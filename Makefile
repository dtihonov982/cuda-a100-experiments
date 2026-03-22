NVCC    ?= nvcc
NVCCFLAGS = -O2 -arch=sm_80

SRCS := $(wildcard *.cu)
BINS := $(SRCS:.cu=)

all: $(BINS)

%: %.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(BINS)

.PHONY: all clean
