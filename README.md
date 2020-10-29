# DLPack.jl

Julia wrapper for [DLPack](https://github.com/dmlc/dlpack).

DLPack provides a data structure to facilitate, sharing the memory of tensor objects
between different deep learning frameworks. Exporting to DLPack is supported in a number
of Python libraries (JAX, CuPy, PyTorch, among others), so this package tries to provide
better interoperability between these libraries and the Julia ecosystem.
