DLPack.jl
---------

[![Tests](https://github.com/pabloferz/DLPack.jl/workflows/CI/badge.svg)](https://github.com/pabloferz/DLPack.jl/actions?query=ci)

Julia wrapper for [DLPack](https://github.com/dmlc/dlpack).

This module provides a Julia interface to facilitate bidirectional data
exchange of tensor objects between Julia and Python libraries such as JAX,
CuPy, PyTorch, among others (all python libraries supporting the
[DLPack protocol][1]).

It can share and wrap CPU and CUDA arrays, and supports interfacing through
both `PyCall` and `PythonCall`.

## Installation

From the Julia REPL activate the package manager (type `]`) and run:

```
pkg> add DLPack
```

## Usage

As an example, let us wrap a JAX array instantiated via the `PyCall` package:

```julia
using DLPack
using PyCall

np = pyimport("jax.numpy")
dl = pyimport("jax.dlpack")

pyv = np.arange(10)
v = from_dlpack(pyv)
# For older jax version use:
# v = DLPack.wrap(pyv, o -> @pycall dl.to_dlpack(o)::PyObject)

(pyv[1] == 1).item()  # This is false since the first element is 0

# Let's mutate an immutable jax DeviceArray
v[1] = 1

(pyv[1] == 1).item()  # true
```

If the python tensor has more than one dimension and the memory layout is
row-major the array returned by `DLPack.from_dlpack` has its dimensions reversed.
Let us illustrate this now by importing a `torch.Tensor` via the
`PythonCall` package:

```julia
using DLPack
using PythonCall

torch = pyimport("torch")

pyv = torch.arange(1, 5).reshape(2, 2)
v = from_dlpack(pyv)
# For older torch releases use:
# v = DLPack.wrap(pyv, torch.to_dlpack)

Bool(v[2, 1] == 2 == pyv[0, 1])  # dimensions are reversed
```

Likewise, we can share Julia arrays to python:

```julia
using DLPack
using PythonCall

torch = pyimport("torch")

v = rand(3, 2)
pyv = DLPack.share(v, torch.from_dlpack)

Bool(pyv.shape == torch.Size((2, 3)))  # again, the dimensions are reversed.
```

Do you want to exchange CUDA tensors? Worry not:

```julia
using DLPack
using CUDA
using PyCall

cupy = pyimport("cupy")

pyv = cupy.arange(6).reshape(2, 3)
v = from_dlpack(pyv)
# For older versions of cupy use:
# v = DLPack.wrap(pyv, o -> pycall(o.toDlpack, PyObject))

v .= 1
pyv.sum().item() == 6  # true

pyw = DLPack.share(v, cupy.from_dlpack)  # new cupy ndarray holding the same data
```

> [!WARNING]
>
> Whenever a Python function allocates a lot of intermediate Python objects, Julia has no
> way of knowing when it should garbage collect such objects, and in some cases the
> allocated memory may grow too large. In such a case, it might be important to manually
> call `GC.gc(false)` from time to time. See
> https://github.com/pabloferz/DLPack.jl/issues/26 for an example of this issue.

[1]: https://data-apis.org/array-api/latest/design_topics/data_interchange.html
