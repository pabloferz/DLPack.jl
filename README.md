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

## Install

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
v = DLPack.wrap(pyv, o -> @pycall dl.to_dlpack(o)::PyObject)

(pyv[1] == 1).item()  # This is false since the first element is 0

# Let's mutate an immutable jax DeviceArray
v[1] = 1

(pyv[1] == 1).item()  # true
```

If the python tensor has more than one dimension and the memory layout is
row-major the array returned by `DLPack.wrap` has its dimensions reversed.
Let us illustrate this now by importing a `torch.Tensor` via the
`PythonCall` package:

```julia
using DLPack
using PythonCall

torch = pyimport("torch")

pyv = torch.arange(1, 5).reshape(2, 2)
v = DLPack.wrap(pyv, dlpack.to_dlpack)

Bool(v[2, 1] == 2 == pyv[0, 1])  # dimensions are reversed
```

Likewise, we can share Julia arrays to python:

```julia
using DLPack
using PythonCall

torch = pyimport("torch")

v = rand(3, 2)
pyv = DLPack.share(v, torch.from_dlpack)

Bool(pyv.shape == torch.Size((2, 3))  # again, the dimensions are reversed.
```

Do you want to exchange CUDA tensors? Worry not:

```julia
using DLPack
using CUDA
using PyCall

cupy = pyimport("cupy")

pyv = cupy.arange(6).reshape(2, 3)
v = DLPack.wrap(pyv, o -> pycall(o.toDlpack, PyObject))

v .= 1
pyv.sum().item() == 6  # true

pyw = DLPack.share(v, cupy.from_dlpack)  # new cupy ndarray holding the same data
```

[1]: https://data-apis.org/array-api/latest/design_topics/data_interchange.html
