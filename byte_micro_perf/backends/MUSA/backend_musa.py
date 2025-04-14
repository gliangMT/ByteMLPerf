import os
import sys
import pathlib
import random
import shutil
import json
from datetime import timedelta

import torch
import torch_musa  # must be imported in MUSA backends
import torch.distributed as dist

FILE_DIR = pathlib.Path(__file__).parent.absolute()
BACKEND_DIR = FILE_DIR.parent
MICRO_PERF_DIR = BACKEND_DIR.parent

sys.path.insert(0, str(MICRO_PERF_DIR))

from core.backend import Backend
from core.utils import suppress_stdout_stderr

# ops
from core.ops.unary_ops import *
from core.ops.binary_ops import *
from core.ops.reduction_ops import *
from core.ops.index_ops import *
from core.ops.ccl_ops import *
from core.ops.h2d_ops import *
from core.ops.gemm_ops import *
from core.ops.attn_ops import *

# TODO(@gliangMT): add custom ops
from .custom_ops import (
    GPUGemmOp,
    GPUGemmFP8Op,
    GPUGroupGemmFP8Op,
    GPUFlashAttentionOp,
    GPUFlashMLAOp,
)


OP_MAPPING = {
    # unary ops
    "cast": CastOp,
    "cos": CosOp,
    "exp": ExpOp,
    "gelu": GeluOp,
    "log": LogOp,
    "silu": SiluOp,
    "sin": SinOp,
    "sqrt": SqrtOp,
    # binary ops
    "add": AddOp,
    "sub": SubOp,
    "mul": MulOp,
    "div": DivOp,
    # reduction ops
    "layer_norm": LayerNormOp,
    "reduce_max": ReduceMaxOp,
    "reduce_sum": ReduceSumOp,
    "reduce_min": ReduceMinOp,
    "softmax": SoftmaxOp,
    "topk": TopkOp,
    # index ops
    "index_select": IndexSelectOp,
    "gather": GatherOp,
    "embedding": EmbeddingOp,
    "scatter": ScatterOp,
    "index_add": IndexAddOp,
    # xccl ops
    "all_gather": AllGatherOp,
    "all_reduce": AllReduceOp,
    "all_to_all": AlltoAllOp,
    # "broadcast": BroadcastOp,
    "reduce_scatter": ReduceScatterOp,
    "p2p": P2POp,
    # h2d ops
    "host2device": Host2DeviceOp,
    "device2host": Device2HostOp,
    "device2device": Device2DeviceOp,
    # TODO(@gliangMT): add custom ops
    # gemm ops
    "gemm": GPUGemmOp,
    "gemm_fp8": GPUGemmFP8Op,
    "group_gemm_fp8": GPUGroupGemmFP8Op,
    # attn ops
    "flash_attention": GPUFlashAttentionOp,
    "flash_mla": GPUFlashMLAOp,
}


class BackendMUSA(Backend):
    def __init__(self):
        super().__init__()

    def __del__(self):
        if self.numa_rank == 0:

            PROFILER_DIR = pathlib.Path.cwd().joinpath("profiling")
            if PROFILER_DIR.exists():
                shutil.rmtree(PROFILER_DIR)

    """
    device management related
    """

    def get_torch_device_name(self):
        return "musa"

    def get_device_name(self, index=0):
        return torch.musa.get_device_name(index)

    def get_device_properties(self, index=0):
        return torch.musa.get_device_properties(index)

    def get_mem_info(self, index=0):
        return torch.musa.mem_get_info(index)

    def get_device_count(self):
        device_count = torch.musa.device_count()
        return device_count, list(range(device_count))

    def set_device(self, device_index: int):
        # some ops need to specify environment variables
        os.environ["MUSA_MEMCPY_PATH"] = "3"
        os.environ["MCCL_PROTOS"] = "2"
        # os.environ["MCCL_ALGOS"]=1         # default
        # export MUSA_BLOCK_SCHEDULE_MODE=1  # default
        os.environ["MCCL_BUFFSIZE"] = "20971520"
        os.environ["MCCL_IB_GID_INDEX"] = "3"
        os.environ["MCCL_NET_SHARED_BUFFERS"] = "0"  # (multi_node alltoall)

        torch.musa.set_device(device_index)

    def get_device(self):
        return torch.musa.current_device()

    def device_synchronize(self):
        torch.musa.synchronize()

    def empty_cache(self):
        torch.musa.empty_cache()

    """
    ccl related
    """

    def get_dist_module(self):
        return dist

    def get_dist_backend(self):
        return "mccl"

    def core_perf(
        self,
        op_instance,
        warmup_iterations,
        prefer_iterations,
        tensor_list,
        profiling=True,
    ):
        op_group = op_instance.op_group
        group_size = op_instance.group_size

        if type(op_instance).is_concurrent() and profiling:
            logger.warning(
                "Profiling is enabled but the operation supports concurrency. "
                "Detailed profiling may not be accurate."
            )

        if not type(op_instance).is_concurrent() and profiling:
            process_id = os.getpid()
            PROFILER_DIR = pathlib.Path.cwd().joinpath("profiling", f"{process_id}")

            # profiling
            with suppress_stdout_stderr():
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.MUSA,
                    ],
                    schedule=torch.profiler.schedule(
                        wait=0,
                        warmup=warmup_iterations,
                        active=prefer_iterations,
                        repeat=1,
                    ),
                ) as prof:
                    for i in range(prefer_iterations + 2):
                        op_instance.core_run(tensor_list[i % len(tensor_list)])
                        self.device_synchronize()
                        prof.step()

            # export profiling results
            torch.profiler.tensorboard_trace_handler(f"{PROFILER_DIR}")(prof)

            # parse and delete profiling json file
            average_latency = 0.0
            kernel_latency_list = {}
            if PROFILER_DIR.exists():
                logger.info(f"Profiling results saved to: {PROFILER_DIR}")
                json_files = list(PROFILER_DIR.glob("*.json"))
                if json_files:
                    profiling_data = json.load(open(json_files[0]))
                    for event in profiling_data["traceEvents"]:
                        if event.get("cat", None) in ["kernel", "gpu_memcpy"]:
                            kernel_name = event["name"]
                            kernel_latency = event["dur"]
                            if kernel_name not in kernel_latency_list:
                                kernel_latency_list[kernel_name] = []
                            kernel_latency_list[kernel_name].append(kernel_latency)

                    take_iters = prefer_iterations // 2
                    iters_offset = prefer_iterations - take_iters

                    removed_keys = []
                    for kernel in kernel_latency_list:
                        if len(kernel_latency_list[kernel]) != prefer_iterations:
                            removed_keys.append(kernel)
                        average_latency += sum(
                            kernel_latency_list[kernel][iters_offset:]
                        )
                    for kernel in removed_keys:
                        kernel_latency_list.pop(kernel)

                    average_latency /= take_iters
                shutil.rmtree(PROFILER_DIR)
            return average_latency, list(kernel_latency_list.keys())

        else:
            for i in range(warmup_iterations):
                index = random.randint(0, len(tensor_list) - 1)
                op_instance.core_run(tensor_list[index])
            start_event = torch.musa.Event(enable_timing=True)
            end_event = torch.musa.Event(enable_timing=True)

            self.device_synchronize()
            self.op_group_barrier(op_group=op_group, group_size=group_size)
            start_event.record()
            for i in range(prefer_iterations):
                op_instance.core_run(tensor_list[i % len(tensor_list)])
            end_event.record()
            end_event.synchronize()

            latency_us = start_event.elapsed_time(end_event) * 1e3 / prefer_iterations
            return latency_us, []
