import os
import json
import csv

import argparse



def read_reports(backend_path:str,repots_path:str):
    root_dir=backend_path
    output_file=repots_path

    if os.path.exists(root_dir):
        print("=================================================================================\n")
        print("                       Successfully found the folder!\n")
        print("=================================================================================\n")
        
    else:
        print("=================================================================================\n")
        print("       This backend reports folder is not exists, Please check the path! \n")
        print(f"                          Default path is ./reports/MTGPU\n")
        print("=================================================================================\n")
        
    
    # 定义需要读取的字段
    fields = ["Operator", "Device Info", "Dtype", "Tensor Shapes",
            "Memory Size(MB)", "Kernel bandwidth(GB/s)", "Avg latency(us)"]

    # 打开CSV文件，准备写入
    with open(output_file, mode='w', newline='') as csv_file:
        # print("open file")
        writer = csv.writer(csv_file)

        # 写入CSV的表头
        writer.writerow(fields)

        # 遍历工作目录
        for subdir, _, files in os.walk(root_dir):

            operator = os.path.basename(subdir)

            for file in files:
                if file.endswith('.json'):
                    file_path = os.path.join(subdir, file)

                    # 读取JSON文件
                    with open(file_path, 'r') as json_file:
                        # print(f"==========json read")
                        data = json.load(json_file)

                        device_info = data.get("Device Info")
                        # print(f"=========={device_info}")

                        # 遍历Performance部分
                        for performance in data["Performance"]:
                            dtype = performance.get("Dtype")
                            tensor_shapes = performance.get("Tensor Shapes")
                            memory_size = performance.get("Memory Size(MB)")
                            err_code = performance.get("Error")
                            if err_code == "OOM":
                                kernel_bandwidth = "OOM"
                                avg_latency = "OOM"
                            else:
                                kernel_bandwidth = performance.get(
                                    "Kernel bandwidth(GB/s)")
                                avg_latency = performance.get("Avg latency(us)")

                            # 写入CSV行
                            writer.writerow(
                                [operator, device_info, dtype, tensor_shapes, memory_size, kernel_bandwidth, avg_latency])

    print(f"Data has been written to {output_file}")

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description='reports reader script')
    parser.add_argument('-i','--input-path',type=str,default="./reports/MTGPU",help="input the backend folder path")
    parser.add_argument('-o','--output-path',type=str,default="reporter_mtgpu.csv",help="output csv file path")
    args=parser.parse_args()
    
    read_reports(args.input_path,args.output_path)
    