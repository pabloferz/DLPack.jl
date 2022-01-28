using CUDA
using DLPack
using PythonCall
using Test

# We rebuild PyCall to use the same python as PythonCall
# otherwise we cannot load them simultaneously
include("rebuild_pycall.jl")
using PyCall


# Load torch with PythonCall
const torch = PythonCall.pyimport("torch")
# Load jax with PyCall
const jax = PyCall.pyimport("jax")
const np = PyCall.pyimport("jax.numpy")
const dlpack = PyCall.pyimport("jax.dlpack")

jax.config.update("jax_enable_x64", true)


@testset "PyCall" begin

    v = np.asarray([1.0, 2.0, 3.0], dtype = np.float32)

    dlv = DLVector{Float32}(@pycall dlpack.to_dlpack(v)::PyObject)
    tensor = dlv.manager.dl_tensor

    @test tensor.ndim == 1 == ndims(dlv)
    @test tensor.dtype == DLPack.jltypes_to_dtypes()[eltype(dlv)]

    if DLPack.device_type(dlv) == DLPack.kDLCPU
        jv = unsafe_wrap(Array, dlv)
        jv[1] = 0  # mutate a jax's tensor
    elseif DLPack.device_type(dlv) == DLPack.kDLGPU
        jv = unsafe_wrap(CuArray, dlv)
        jv[1:1] .= 0  # mutate a jax's tensor
    end

    @test py"$np.all($v[:] == $np.asarray([0.0, 2.0, 3.0])).item()"

    w = np.asarray([1 2; 3 4], dtype = np.int64)

    dlw = DLArray(@pycall dlpack.to_dlpack(w)::PyObject)
    tensw = dlw.manager.dl_tensor

    @test tensw.ndim == 2 == ndims(dlw)
    @test tensw.dtype == DLPack.jltypes_to_dtypes()[eltype(dlw)]

    if DLPack.device_type(dlw) == DLPack.kDLCPU
        jw = unsafe_wrap(Array, dlw)
        @test jw[2, 1] == 2
    elseif DLPack.device_type(dlw) == DLPack.kDLCUDA
        jw = unsafe_wrap(CuArray, dlw)
        @test all(view(jw, 2, 1) .== 2)
    end

end


@testset "PythonCall" begin

    v = torch.ones((2, 4), dtype = torch.float64)

    dlv = DLMatrix{Float64}(torch.to_dlpack(v))
    tensor = dlv.manager.dl_tensor

    @test tensor.ndim == 2 == ndims(dlv)
    @test tensor.dtype == DLPack.jltypes_to_dtypes()[eltype(dlv)]

    if DLPack.device_type(dlv) == DLPack.kDLCPU
        jv = unsafe_wrap(Array, dlv)
        jv[1] = 0  # mutate a jax's tensor
    elseif DLPack.device_type(dlv) == DLPack.kDLGPU
        jv = unsafe_wrap(CuArray, dlv)
        jv[1:1] .= 0  # mutate a jax's tensor
    end

    ref = torch.tensor(((0, 1, 1, 1), (1, 1, 1, 1)), dtype = torch.float64)
    @test Bool(torch.all(v == ref).item())

end
