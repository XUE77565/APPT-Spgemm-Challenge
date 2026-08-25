#!/bin/bash
# GPU 健康预检(2026-08-25 起,长跑批/计时前必跑)—— 7-27/7-30 与 8-25 三次 wedge 的教训
#   用法:bash scripts/gpu_check.sh [GPU_ID]   (默认查全部)
#   检查:①nvidia-smi 可应答且无 "requires reset" ②1-thread kernel + D2H 冒烟
#   退出码:0 = 健康;1 = 异常(绝不在该卡上跑任何东西)
set -u
GPUS="${1:-}"

out=$(nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader 2>&1)
if [ $? -ne 0 ]; then echo "✗ nvidia-smi 无响应(驱动可能挂死)"; exit 1; fi
if [ -n "$GPUS" ]; then out=$(echo "$out" | awk -F, -v g="$GPUS" '$1 ~ ("^("g")$") || index(g, $1)'); fi
if echo "$out" | grep -qE "\[N/A\]|requires reset|ERR"; then
  echo "✗ GPU 状态异常:"; echo "$out"; exit 1
fi
echo "✓ nvidia-smi 正常:$(echo "$out" | tr '\n' ' ')"

# 微型 CUDA 冒烟:1-thread kernel + D2H(编译一次,缓存复用)
SMOKE=/tmp/gpu_smoke_check
src=/tmp/gpu_smoke.cu
if [ ! -x "$SMOKE" ] || [ "$src" -nt "$SMOKE" ]; then
  cat > "$src" <<'EOF'
#include <cstdio>
__global__ void k(int *o) { if (threadIdx.x == 0) *o = 42; }
int main() {
    int *d, h = 0;
    if (cudaMalloc(&d, 4) != cudaSuccess) return 1;
    k<<<1, 32>>>(d);
    if (cudaDeviceSynchronize() != cudaSuccess) return 2;
    if (cudaMemcpy(&h, d, 4, cudaMemcpyDeviceToHost) != cudaSuccess) return 3;
    printf("%s\n", h == 42 ? "SMOKE_OK" : "WRONG_VALUE");
    return h == 42 ? 0 : 4;
}
EOF
  /usr/local/cuda/bin/nvcc -O1 -arch=sm_90 "$src" -o "$SMOKE" 2>/dev/null || { echo "✗ 冒烟程序编译失败"; exit 1; }
fi
for g in $GPUS $( [ -z "$GPUS" ] && nvidia-smi --list-gpus | sed 's/.*GPU \([0-9]*\).*/\1/' ); do
  [ -z "$g" ] && continue
  r=$(CUDA_VISIBLE_DEVICES=$g timeout 30 "$SMOKE" 2>&1)
  if [ "$r" = "SMOKE_OK" ]; then echo "✓ GPU $g 冒烟通过"
  else echo "✗ GPU $g 冒烟失败($r)——禁止在该卡运行"; exit 1; fi
done
exit 0
