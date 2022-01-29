using CUDA
using DLPack
using PythonCall
using Test

# We rebuild PyCall to use the same python as PythonCall
# otherwise we cannot load them simultaneously
include("rebuild_pycall.jl")
using PyCall


# Just for the sake of variety:
# Load torch with PythonCall
const torch = PythonCall.pyimport("torch")
# Load jax with PyCall
const jax = PyCall.pyimport("jax")
const np = PyCall.pyimport("jax.numpy")
const dlpack = PyCall.pyimport("jax.dlpack")

jax.config.update("jax_enable_x64", true)


@testset "PyCall" begin
    to_dlpack = o -> @pycall dlpack.to_dlpack(o)::PyObject

    v = np.asarray([1.0, 2.0, 3.0], dtype = np.float32)
    dlv = DLArray(v, to_dlpack)
    opaque_tensor = dlv.manager.tensor.dl_tensor

    @test v.ndim == 1 == ndims(dlv)
    @test opaque_tensor.dtype == DLPack.jltypes_to_dtypes()[eltype(dlv)]

    if DLPack.device_type(opaque_tensor) == DLPack.kDLCPU
        dlv.data[1] = 0  # mutate a jax's tensor
        @inferred DLVector{Float32}(Array, ColMajor, v, to_dlpack)
    elseif DLPack.device_type(opaque_tensor) == DLPack.kDLGPU
        dlv.data[1:1] .= 0  # mutate a jax's tensor
        @inferred DLVector{Float32}(CuArray, ColMajor, v, to_dlpack)
    end

    @test py"$np.all($v[:] == $np.asarray([0.0, 2.0, 3.0])).item()"

    w = np.asarray([1 2; 3 4], dtype = np.int64)
    dlw = DLArray(w, to_dlpack)
    opaque_tensor = dlw.manager.tensor.dl_tensor

    @test w.ndim == 2 == ndims(dlw)
    @test opaque_tensor.dtype == DLPack.jltypes_to_dtypes()[eltype(dlw)]

    if DLPack.device_type(opaque_tensor) == DLPack.kDLCPU
        @test dlw.data[2, 1] == 3
    elseif DLPack.device_type(opaque_tensor) == DLPack.kDLCUDA
        @test all(view(dlw.data, 2, 1) .== 3)
    end

end


@testset "PythonCall" begin

    v = torch.ones((2, 4), dtype = torch.float64)
    dlv = DLArray(v, torch.to_dlpack)
    opaque_tensor = dlv.manager.tensor.dl_tensor

    @test pyconvert(Int,opaque_tensor.ndim) == 2 == ndims(dlv)
    @test opaque_tensor.dtype == DLPack.jltypes_to_dtypes()[eltype(dlv)]
    @test dlv.data isa PermutedDimsArray

    if DLPack.device_type(opaque_tensor) == DLPack.kDLCPU
        dlv.data[2] = 0  # mutate a jax's tensor
        @inferred DLMatrix{Float64}(Array, RowMajor, v, torch.to_dlpack)
    elseif DLPack.device_type(opaque_tensor) == DLPack.kDLGPU
        dlv.data[2:2] .= 0  # mutate a jax's tensor
        @inferred DLMatrix{Float64}(CuArray, RowMajor, v, torch.to_dlpack)
    end

    ref = torch.tensor(((1, 1, 1, 1), (0, 1, 1, 1)), dtype = torch.float64)
    @test Bool(torch.all(v == ref).item())

end
