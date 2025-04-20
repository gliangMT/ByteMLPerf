import sys
import pathlib
import torch
import torch_musa
import random
import time

from typing import List, Dict, Union, Tuple

FILE_DIR = pathlib.Path(__file__).parent.absolute()
MICRO_PERF_DIR = FILE_DIR.parent.parent

ITERS = 1000

sys.path.insert(0, str(MICRO_PERF_DIR))

from core.utils import logger
from core.utils import OpTensorInfo, calc_tensor_size
from core.op import BasicOp
from core.ops.gemm_ops import GemmOp, GemmFP8Op, GroupGemmFP8Op
from core.ops.attn_ops import FlashAttentionOp

# group fp8 gemm
from transformer_engine.pytorch.cpp_extensions import general_gemm, general_grouped_gemm
from transformer_engine.pytorch.tensor.float8_tensor import Float8Quantizer
from transformer_engine.pytorch.module.base import (
    get_multi_stream_cublas_workspace,
    get_workspace,
)
import transformer_engine_torch as tex

"""
gemm ops
"""


class GPUGemmOp(GemmOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

        if self.dtype == "float32":
            torch.backends.mudnn.allow_tf32 = False
        elif self.dtype == "tfloat32":
            torch.backends.mudnn.allow_tf32 = True


FP8_E4M3_MAX = 448.0  # Maximum representable value in FP8 E4M3 format


class GPUGemmFP8Op(GemmFP8Op):
    def __init__(self, args_dict, backend, *args, **kwargs):

        super().__init__(args_dict, backend, *args, **kwargs)

        self._custom_run = True
        self._run_func = self.gemm_fp8_run

    def test_scaled_mm_with_cast(self):
        """
        c = a @ b -> c8 = a8 @ b8
        c8 = f8(c * scale_out) = f8(a * scale_a) @ f8(b * scale_b) = f8[scale_a * scale_b * (a @ b)]
        """
        m = self.M
        k = self.K
        n = self.N
        device = self.backend.get_torch_device_name()
        mat1 = torch.randn((m, k), dtype=torch.float32)
        mat2 = torch.randn((k, n), dtype=torch.float32)

        fp8max = FP8_E4M3_MAX
        # get f8 tensor from inputs
        scale_a = mat1.abs().max() / fp8max
        scale_b = mat2.abs().max() / fp8max
        f8_a = (mat1 / scale_a).to(torch.float8_e4m3fn)
        f8_b = (mat2 / scale_b).to(torch.float8_e4m3fn)

        # fp32 golden result
        golden = torch.mm(f8_a.float(), f8_b.float()) * scale_a * scale_b

        # out_dtype scaled_mm result
        scale_out = golden.abs().max() / fp8max

        f8_a_gpu = f8_a.to(device)
        f8_b_gpu = f8_b.to(device)
        scale_a_gpu = scale_a.to(device)
        scale_b_gpu = scale_b.to(device)
        scale_out_gpu = scale_out.to(device)

        self.backend.device_synchronize()
        # warm up
        musa_out, amax = torch._scaled_mm(
            f8_a_gpu,
            f8_b_gpu,
            scale_a=scale_a_gpu,
            scale_b=scale_b_gpu,
            scale_result=scale_out_gpu,
            out_dtype=torch.bfloat16,
        )

        self.backend.device_synchronize()
        start_time = time.perf_counter()
        for _ in range(ITERS):
            musa_out, amax = torch._scaled_mm(
                f8_a_gpu,
                f8_b_gpu,
                scale_a=scale_a_gpu,
                scale_b=scale_b_gpu,
                scale_result=scale_out_gpu,
                out_dtype=torch.bfloat16,
            )
        self.backend.device_synchronize()
        end_time = time.perf_counter()

        exec_time = end_time - start_time
        return (exec_time / ITERS) * 1e6  # us

    def gemm_fp8_run(self):
        return self.test_scaled_mm_with_cast()


class GPUGroupGemmFP8Op(GroupGemmFP8Op):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

        self._custom_run = True
        self._run_func = self.group_gemm_fp8_run

    def test_fp8_grouped_gemm(self, accumulate=True):
        device = self.backend.get_torch_device_name()
        # z, m, k, n = shape
        z = self.num_groups
        m = self.M
        k = self.K
        n = self.N
        m_splits = m // z

        dtype = torch.bfloat16
        A = [torch.randn(n, k, dtype=dtype, device=device) for _ in range(z)]  # weight
        B = torch.split(
            torch.randn(m, k, dtype=dtype, device=device), m_splits
        )  # input
        out = torch.split(
            torch.randn(m, n, dtype=dtype, device=device), m_splits
        )  # output
        out_ref = [o.clone() for o in out]

        # fp8 should be robust enough to this fake scale
        scale = 1 + torch.rand(1, dtype=torch.float32, device=device).squeeze()
        amax = torch.zeros(1, 1, dtype=torch.float32, device=device)

        a_quantizers = [
            Float8Quantizer(
                scale.clone(),
                amax.clone(),
                tex.DType.kFloat8E4M3,
            )
            for _ in range(z)
        ]
        b_quantizers = [
            Float8Quantizer(
                scale.clone(),
                amax.clone(),
                tex.DType.kFloat8E4M3,
            )
            for _ in range(z)
        ]

        A_fp8 = []
        B_fp8 = []

        for i in range(z):
            A_fp8.append(a_quantizers[i](A[i]))
            B_fp8.append(b_quantizers[i](B[i]))

        # warm up
        self.backend.device_synchronize()
        general_grouped_gemm(
            A_fp8,
            B_fp8,
            out,
            dtype,
            get_multi_stream_cublas_workspace(),
            m_splits=[k] * m_splits,
            accumulate=accumulate,
        )

        self.backend.device_synchronize()
        start_time = time.perf_counter()
        for _ in range(ITERS):
            general_grouped_gemm(
                A_fp8,
                B_fp8,
                out,
                dtype,
                get_multi_stream_cublas_workspace(),
                m_splits=[k] * m_splits,
                accumulate=accumulate,
            )
        end_time = time.perf_counter()
        self.backend.device_synchronize()
        exec_time = end_time - start_time

        tflops = 2 * m * k * n * z / (exec_time * 1e12)
        print(f">>>>>>>>>> actual calc flops is {tflops} ")

        return (exec_time / ITERS) * 1e6  # us

    def group_gemm_fp8_run(self):
        return self.test_fp8_grouped_gemm()


"""
attn_ops
"""

try:
    import flash_attn
    from flash_attn import flash_attn_func
except ImportError:
    flash_attn = None

try:
    # import flash_attn_interface
    # from flash_attn_interface import flash_attn_func
    from flash_attn import flash_attn_interface
    from flash_attn.flash_attn_interface import flash_attn_func
except ImportError:
    flash_attn_interface = None


class GPUFlashAttentionOp(FlashAttentionOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        if flash_attn is None and flash_attn_interface is None:
            raise ImportError(
                "flash_attention is not available, please install it first."
            )

        super().__init__(args_dict, backend, *args, **kwargs)
        self._run_func = self.flash_attention_run

        # create output tensor during testing
        self.output_tensor_info = {}

    def flash_attention_run(self, tensor_mapping):
        q = tensor_mapping["q"]
        k = tensor_mapping["k"]
        v = tensor_mapping["v"]
        return flash_attn_func(q, k, v, causal=self.is_causal)


try:
    import flash_mla
    from flash_mla import flash_mla_with_kvcache, get_mla_metadata
except ImportError:
    flash_mla = None
    # logger.warning("flash_mla is not available, please install it first.")


class GPUFlashMLAOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        if flash_mla is None or flash_attn_interface is None:
            raise ImportError(
                "flash_mla or flash_attn is not available, please install it first."
            )

        super().__init__(args_dict, backend, *args, **kwargs)
        self._run_func = self.flash_mla_run

    def prepare(self):
        # llm args
        self.arg_type = self.args_dict["arg_type"]
        if self.arg_type != "llm":
            raise NotImplementedError

        # llm phase: prefill or decode
        self.phase = self.args_dict["phase"]
        if self.phase not in ["prefill", "decode"]:
            raise NotImplementedError

        # dtype: bfloat16
        self.dtype = self.args_dict["dtype"]
        if self.dtype != "bfloat16":
            raise NotImplementedError
        self.torch_dtype = torch.bfloat16
        self.torch_dtype_size = torch.tensor([], dtype=self.torch_dtype).element_size()

        self.batch_size = self.args_dict["batch_size"]
        self.q_seq_len = self.args_dict["q_seq_len"]
        self.kv_seq_len = self.args_dict["kv_seq_len"]
        self.q_head_num = self.args_dict["q_head_num"]
        self.kv_head_num = self.args_dict["kv_head_num"]
        self.qk_dim_size = self.args_dict["qk_dim_size"]
        self.v_dim_size = self.args_dict["v_dim_size"]

        self.is_causal = self.args_dict["is_causal"]
        if not self.is_causal:
            raise NotImplementedError

        self.varlen = self.args_dict["varlen"]
        if self.varlen:
            raise NotImplementedError

        # q: [batch_size, q_seq_len, q_head_num, qk_dim_size]
        self.q = torch.randn(
            self.batch_size,
            self.q_seq_len,
            self.q_head_num,
            self.qk_dim_size,
            dtype=self.torch_dtype,
            device=self.backend.get_torch_device_name(),
        )

        # prefill, not absorb weight, use flash_attention
        if self.phase == "prefill":
            self.k = torch.randn(
                self.batch_size,
                self.kv_seq_len,
                self.kv_head_num,
                self.qk_dim_size,
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
            )
            self.v = torch.randn(
                self.batch_size,
                self.kv_seq_len,
                self.kv_head_num,
                self.qk_dim_size,
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
            )

            self.input_tensor_size = {
                "q": OpTensorInfo(
                    shape=[
                        self.batch_size,
                        self.q_seq_len,
                        self.q_head_num,
                        self.qk_dim_size,
                    ],
                    dtype=self.torch_dtype,
                    device=self.backend.get_torch_device_name(),
                ),
                "k": OpTensorInfo(
                    shape=[
                        self.batch_size,
                        self.kv_seq_len,
                        self.kv_head_num,
                        self.qk_dim_size,
                    ],
                    dtype=self.torch_dtype,
                    device=self.backend.get_torch_device_name(),
                ),
                "v": OpTensorInfo(
                    shape=[
                        self.batch_size,
                        self.kv_seq_len,
                        self.kv_head_num,
                        self.qk_dim_size,
                    ],
                    dtype=self.torch_dtype,
                    device=self.backend.get_torch_device_name(),
                ),
            }
            self.output_tensor_size = {
                "out": OpTensorInfo(
                    shape=[
                        self.batch_size,
                        self.q_seq_len,
                        self.q_head_num,
                        self.qk_dim_size,
                    ],
                    dtype=self.torch_dtype,
                    device=self.backend.get_torch_device_name(),
                )
            }

            self.input_tensor_size = sum(
                [calc_tensor_size(info) for info in self.input_tensor_size.values()]
            )
            self.output_tensor_size = sum(
                [calc_tensor_size(info) for info in self.output_tensor_size.values()]
            )
            self.tensor_size = self.input_tensor_size + self.output_tensor_size

            self.read_bytes = self.input_tensor_size
            self.write_bytes = self.output_tensor_size
            self.io_bytes = self.read_bytes + self.write_bytes

            self.algo_size = 0
            self.bus_size = 0

            self.attn_ratio = (1 + self.kv_seq_len) / 2 / self.kv_seq_len
            self.calc_flops = (
                self.batch_size
                * self.q_head_num
                * self.q_seq_len
                * self.kv_seq_len
                * (self.qk_dim_size + self.v_dim_size)
                * 2
                * self.attn_ratio
            )

        # decode, absorb weight, use flash_mla
        elif self.phase == "decode":
            self.cache_seqlens = torch.full(
                (self.batch_size,),
                self.kv_seq_len,
                dtype=torch.int32,
                device=self.backend.get_torch_device_name(),
            )
            self.total_seqlens = self.cache_seqlens.sum().item()
            self.mean_seqlens = self.cache_seqlens.float().mean().int().item()
            self.max_seqlen = self.cache_seqlens.max().item()
            self.max_seqlen_pad = (self.max_seqlen + 255) // 256 * 256

            self.block_size = 64
            self.block_table = torch.arange(
                self.batch_size * self.max_seqlen_pad // self.block_size,
                dtype=torch.int32,
                device=self.backend.get_torch_device_name(),
            ).view(self.batch_size, self.max_seqlen_pad // self.block_size)

            self.blocked_k = torch.randn(
                self.block_table.numel(),
                self.block_size,
                self.kv_head_num,
                self.qk_dim_size,
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
            )
            for i in range(self.batch_size):
                self.blocked_k.view(
                    self.batch_size,
                    self.max_seqlen_pad,
                    self.kv_head_num,
                    self.qk_dim_size,
                )[i, self.cache_seqlens[i].item() :] = float("nan")
            self.tile_scheduler_metadata, self.num_splits = get_mla_metadata(
                self.cache_seqlens,
                self.q_seq_len * self.q_head_num // self.kv_head_num,
                self.kv_head_num,
            )

            # q:            [batch_size, q_seq_len, q_head_num, qk_dim_size]
            # blocked_k:    [batch_size * max_seqlen_pad // block_size, block_size, kv_head_num, qk_dim_size]
            # block_table:  [batch_size, max_seqlen_pad // block_size]
            # cache_seqlens:[batch_size]
            self.input_tensor_size = {
                "q": OpTensorInfo(
                    shape=[
                        self.batch_size,
                        self.q_seq_len,
                        self.q_head_num,
                        self.qk_dim_size,
                    ],
                    dtype=self.torch_dtype,
                    device=self.backend.get_torch_device_name(),
                ),
                "blocked_k": OpTensorInfo(
                    shape=[
                        self.block_table.numel(),
                        self.block_size,
                        self.kv_head_num,
                        self.qk_dim_size,
                    ],
                    dtype=self.torch_dtype,
                    device=self.backend.get_torch_device_name(),
                ),
                "block_table": OpTensorInfo(
                    shape=[self.batch_size, self.max_seqlen_pad // self.block_size],
                    dtype=torch.int32,
                    device=self.backend.get_torch_device_name(),
                ),
                "cache_seqlens": OpTensorInfo(
                    shape=[self.batch_size],
                    dtype=torch.int32,
                    device=self.backend.get_torch_device_name(),
                ),
            }

            # out:          [batch_size, q_seq_len, q_head_num, v_dim_size]
            # softmax_lse   [batch_size, q_head_num, q_seq_len]
            self.output_tensor_size = {
                "out": OpTensorInfo(
                    shape=[
                        self.batch_size,
                        self.q_seq_len,
                        self.q_head_num,
                        self.v_dim_size,
                    ],
                    dtype=self.torch_dtype,
                    device=self.backend.get_torch_device_name(),
                ),
                "softmax_lse": OpTensorInfo(
                    shape=[self.batch_size, self.q_head_num, self.q_seq_len],
                    dtype=torch.float32,
                    device=self.backend.get_torch_device_name(),
                ),
            }
            self.input_tensor_size = sum(
                [calc_tensor_size(info) for info in self.input_tensor_size.values()]
            )
            self.output_tensor_size = sum(
                [calc_tensor_size(info) for info in self.output_tensor_size.values()]
            )
            self.tensor_size = self.input_tensor_size + self.output_tensor_size

            # q + kv_compress, ignore block_table and cache_seqlens
            self.read_bytes = (
                self.batch_size * self.q_seq_len * self.q_head_num * self.qk_dim_size
                + self.total_seqlens * self.kv_head_num * self.qk_dim_size
            ) * self.torch_dtype_size
            # out + softmax_lse
            self.write_bytes = self.output_tensor_size
            self.io_bytes = self.read_bytes + self.write_bytes

            self.algo_size = 0
            self.bus_size = 0

            # q * k, p * v
            self.calc_flops = (
                self.total_seqlens
                * self.q_head_num
                * self.q_seq_len
                * (self.qk_dim_size + self.v_dim_size)
                * 2
            )

    def create_tensors(self, instance_num: int):
        all_tensor_list = []
        for i in range(instance_num):
            tensor_mapping = {}
            if self.phase == "prefill":
                tensor_mapping["q"] = self.q.clone()
                tensor_mapping["k"] = self.k.clone()
                tensor_mapping["v"] = self.v.clone()
            elif self.phase == "decode":
                tensor_mapping["q"] = self.q.clone()
                tensor_mapping["blocked_k"] = self.blocked_k.clone()
                tensor_mapping["block_table"] = self.block_table.clone()
                tensor_mapping["cache_seqlens"] = self.cache_seqlens.clone()
            all_tensor_list.append(tensor_mapping)
        return all_tensor_list

    @torch.inference_mode()
    def flash_mla_run(self, tensor_mapping):
        if self.phase == "prefill":
            q = tensor_mapping["q"]
            k = tensor_mapping["k"]
            v = tensor_mapping["v"]
            return flash_attn_func(q, k, v, causal=self.is_causal)
        elif self.phase == "decode":
            q = tensor_mapping["q"]
            blocked_k = tensor_mapping["blocked_k"]
            block_table = tensor_mapping["block_table"]
            cache_seqlens = tensor_mapping["cache_seqlens"]
            return_vals = flash_mla_with_kvcache(
                q,
                blocked_k,
                block_table,
                cache_seqlens,
                self.v_dim_size,
                self.tile_scheduler_metadata,
                self.num_splits,
                causal=self.is_causal,
            )
            return return_vals
