NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_90 -std=c++14
INCLUDES = -Iinclude -I/usr/local/cuda/include
LIBS = -lcusparse -lcublas

TARGET = spgemm_test
SRCS = src/main.cu src/matrix_utils.cu src/spgemm_kernel_cusparse.cu src/spgemm_kernel_manual.cu
OBJS = $(SRCS:.cu=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(LIBS)

src/%.o: src/%.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

# 声明 .o 为中间文件,make 完成后会自动删除它们
.INTERMEDIATE: $(OBJS)

clean:
	rm -f src/*.o $(TARGET) *.mtx
	rm -rf results/*

test: $(TARGET)
	./$(TARGET) data/sphere2/sphere2.mtx

run_all: $(TARGET)
	bash run_all.sh

.PHONY: all clean test run_all