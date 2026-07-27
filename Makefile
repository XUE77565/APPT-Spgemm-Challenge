NVCC = nvcc
# DBG=1 开启 [dbg ms][tag] phase 时间戳(供 compute-only 解析,排除 h2d/d2h,对标 full_compare)
NVCC_FLAGS = -O3 -arch=sm_90 -std=c++14 $(if $(DBG),-DDBG=1) $(if $(WRITE_MTX),-DWRITE_MTX=1)
INCLUDES = -Iinclude -I/usr/local/cuda/include
LIBS = -lcusparse -lcublas

TARGET = spgemm_test
SRCS = src/main.cu src/matrix_utils.cu src/spgemm_kernel_cusparse.cu src/spgemm_kernel_manual.cu src/spgemm_kernel_formulations.cu src/spgemm_kernel_hash.cu src/spgemm_adaptive.cu src/spgemm_merge.cu src/mempool.cu
OBJS = $(SRCS:.cu=.o)
HEADERS = $(wildcard include/*.h)   # 任何头文件改动都触发重编

all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(LIBS)

src/%.o: src/%.cu $(HEADERS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

# 声明 .o 为中间文件,make 完成后会自动删除它们
.INTERMEDIATE: $(OBJS)

clean:
	rm -f src/*.o $(TARGET) *.mtx
	rm -rf results/*

test: $(TARGET)
	./$(TARGET) data/sphere2/sphere2.mtx

# 稠密 baseline 算子(dense-for-sparse:densify + cuBLAS dgemm + sparsify,全 GPU compute)
# cudaEvent 计 densify+dgemm+sparsify(不含 h2d/d2h)= 与 hash/cuSPARSE compute-only 同口径
spgemm_dense: src/spgemm_dense.cu src/matrix_utils.cu src/mempool.cu $(HEADERS)
	$(NVCC) -O3 -arch=sm_90 -std=c++14 $(INCLUDES) -o $@ src/spgemm_dense.cu src/matrix_utils.cu src/mempool.cu $(LIBS)

dense: spgemm_dense

run_all: $(TARGET)
	bash scripts/run_all.sh

.PHONY: all clean test run_all