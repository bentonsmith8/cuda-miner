NVCC        = nvcc
NVCC_FLAGS  = -O3 -I/usr/local/cuda/include  
LD_FLAGS    = -lcudart -L/usr/local/cuda/lib64
EXE			= shatester
LIB 		= libfindhash.so

default: library

library:
	nvcc --ptxas-options=-v --compiler-options '-fPIC -O3' -o $(LIB) --shared findhash.cu

shatester.o: shatester.cu
	$(NVCC) -c -o $@ shatester.cu $(NVCC_FLAGS)

findhash.o: findhash.cu
	$(NVCC) -c -o $@ findhash.cu $(NVCC_FLAGS)

$(EXE): shatester.o findhash.o
	$(NVCC) shatester.o findhash.o -o $(EXE) $(LD_FLAGS)

clean:
	rm -rf *.o *.so $(EXE)