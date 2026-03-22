NVCC      ?= nvcc
NVCCFLAGS  = -O2 -arch=sm_80
BUILDDIR   = build

SRCS := $(wildcard *.cu)
BINS := $(patsubst %.cu,$(BUILDDIR)/%,$(SRCS))

all: $(BUILDDIR) $(BINS)

$(BUILDDIR):
	mkdir -p $(BUILDDIR)

$(BUILDDIR)/%: %.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -rf $(BUILDDIR)

.PHONY: all clean
