import torch
import fbgemm_gpu
from benchmark_utils import GPUTimer

import numpy as np
import matplotlib.pyplot as plt

def translateToPowerLaw(min, max, alpha, x):
    gamma = torch.tensor([1 - alpha], device=x.device)
    y = torch.pow(
        x * (torch.pow(max, gamma) - torch.pow(min, gamma)) + torch.pow(min, gamma),
        1.0 / gamma,
    )
    b = y >= max
    y[b] = max - 1
    return y


def PowerLaw(min, max, alpha, N, device=torch.device("cuda"), permute=None):
    x = torch.rand(N, device=device, dtype=torch.float64)
    y = translateToPowerLaw(min, max, alpha, x).to(torch.int64)

    if permute != None:
        y = permute[y]

    return y


def gen_key(batch, hotness, alpha, N, device, permute=None):
    ret = PowerLaw(1, N, alpha, hotness * batch, device, permute)
    return ret

def plot(datas, labels, num_blocks, file):
    fig, ax = plt.subplots(figsize=(128, 4)) 
    # 通常无需特别设置 auto aspect，或可省略，除非特殊比例需求
    # ax.set_aspect('auto')
    for data, label in zip(datas, labels):
        ax.plot(data, label=label)
    
    # 自动根据第一组数据长度设置横轴范围
    x_len = len(datas) if datas else 0
    # plt.xticks(list(range(0, x_len, max(1, x_len // 10))))
    plt.xlabel('block id')
    plt.ylabel('Latency(H2D/ms)')
    fig.suptitle(f'Total blocks={num_blocks}')
    ax.legend()  # 显示图例
    
    plt.savefig(f'./{file}.png', transparent=False)  # 保存图像
    plt.close(fig)  # 推荐关闭 fig 实例

    


def test():
  assert torch.cuda.is_available()
  size = 1024 ** 3
  dtype = torch.float32
  # uvm_host_mapped (bool = False): If True, allocate every UVM tensor
  #     using `malloc` + `cudaHostRegister`. Otherwise use
  #     `cudaMallocManaged`
  uvm_host_mapped = False
  current_device = torch.cuda.current_device()
  x = torch.zeros(
      size,
      out=torch.ops.fbgemm.new_unified_tensor(
          # pyre-fixme[6]: Expected `Optional[Type[torch._dtype]]`
          #  for 3rd param but got `Type[Type[torch._dtype]]`.
          torch.zeros(1, device=current_device, dtype=dtype),
          [size],
          is_host_mapped=uvm_host_mapped,
      ),
  )
  out=torch.ops.fbgemm.new_unified_tensor(
      # pyre-fixme[6]: Expected `Optional[Type[torch._dtype]]`
      #  for 3rd param but got `Type[Type[torch._dtype]]`.
      torch.zeros(1, device=current_device, dtype=dtype),
      [size],
      is_host_mapped=uvm_host_mapped,
  )
  print(type(x), x.size(), x.device)
  print(type(out), out.size(), out.device)
  print(x[0].item(), out[0].item())

def warmup_gpu(device="cuda"):
    # 1. compute unit
    a = torch.randn(10, 16384, 2048, device=device)
    b = torch.randn(10, 2048, 16384, device=device)
    for _ in range(5):
        torch.matmul(a, b)
        torch.cuda.synchronize()

    # 2. copy engine
    d_cpu = torch.randn(10, 1024, 1024)
    d_gpu = torch.empty_like(d_cpu, device=device)
    for _ in range(5):
        # CPU -> GPU
        d_gpu.copy_(d_cpu, non_blocking=True)
        torch.cuda.synchronize()
        # GPU -> CPU
        d_cpu.copy_(d_gpu, non_blocking=True)
        torch.cuda.synchronize()

def benchmark_page_size(size_per_iter, total_bytes, file, stride=1):
    dtype=torch.float32
    current_device = torch.cuda.current_device()
    dim = size_per_iter // 4
    num_rows = total_bytes // size_per_iter
    x = torch.ops.fbgemm.new_unified_tensor(
        torch.zeros(1, device=current_device, dtype=dtype),
        [num_rows, dim],
        is_host_mapped=True,
    )
    y = torch.ops.fbgemm.new_unified_tensor(
        torch.zeros(1, device=current_device, dtype=dtype),
        [num_rows, dim],
        is_host_mapped=False,
    )
    
    x_out = torch.empty(dim, dtype=dtype, device=current_device)
    y_out = torch.empty(dim, dtype=dtype, device=current_device)
    timer = GPUTimer()
    x_perf = []
    for i in range(0, num_rows * 2, stride):
        timer.start()
        x_out.copy_(x[i % num_rows,:], non_blocking=True)
        timer.stop()
        x_perf.append(timer.elapsed_time())

    y_perf = []
    for i in range(0, num_rows * 2, stride):
        timer.start()
        y_out.copy_(y[i % num_rows,:], non_blocking=True)
        timer.stop()
        y_perf.append(timer.elapsed_time())
    
    for i in range(len(x_perf)):
        print(f"{i}: {x_perf[i]}, {y_perf[i]}")
    
    plot([x_perf, y_perf], ['host_mapped=true', 'host_mapped=false'], num_rows, file)
    
        
  
def benchmark(num_rows, batch, dim=128, dtype=torch.float32, alpha=1.05):
    current_device = torch.cuda.current_device()
    x = torch.ops.fbgemm.new_unified_tensor(
        torch.zeros(1, device=current_device, dtype=dtype),
        [num_rows, dim],
        is_host_mapped=True,
    )
    y = torch.ops.fbgemm.new_unified_tensor(
        torch.zeros(1, device=current_device, dtype=dtype),
        [num_rows, dim],
        is_host_mapped=False,
    )
    # print(type(x), x.size(), x.device)
    indices_list = []
    repeat = 20
    indices = gen_key(batch * 20, 1, alpha, num_rows, current_device)
    unique_indices_list = []
    for i in range(repeat):
        indices_slice = indices[batch*i : batch*(i+1)]
        unique_indices = torch.unique(indices_slice, sorted=False,)
        print(indices_slice)
        unique_indices_list.append(indices_slice)
    
    timer = GPUTimer()
    
    warmup_gpu()

    timer.start()
    for i in range(20):
        out_y = y[unique_indices_list[i]].to(current_device)
    timer.stop()
    print(f"UVM overhead: {timer.elapsed_time()} ms")
    
    timer.start()
    for i in range(20):
        out_x = x[unique_indices_list[i]].to(current_device)
    timer.stop()
    print(f"UVM host mapped overhead: {timer.elapsed_time()} ms")
    
    timer.start()
    for i in range(20):
        out_y = y[unique_indices_list[i]].to(current_device)
    timer.stop()
    print(f"UVM overhead: {timer.elapsed_time()} ms")
    
    timer.start()
    for i in range(20):
        out_x = x[unique_indices_list[i]].to(current_device)
    timer.stop()
    print(f"UVM host mapped overhead: {timer.elapsed_time()} ms")


if __name__ == "__main__":
    #test()
    #benchmark_page_size(512 * 1024, 8 * 1024 * 1024, 'test_page_size')
    benchmark_page_size(512 * 1024, 32 * 1024 * 1024 * 1024, 'test_TLB_size1', stride=64)
    benchmark_page_size(512 * 1024, 32 * 1024 * 1024 * 1024, 'test_TLB_size2', stride=32)
    
    # benchmark_page_size(128 * 1024 * 1024, 8 * 1024 * 1024 * 1024, 'test_TLB_size')
    # benchmark_page_size(128 * 1024 * 1024, 8 * 1024 * 1024 * 1024, 'test_TLB_size')
    
    # benchmark(64*1024*1024, 1024* 1024, alpha=1.05)
