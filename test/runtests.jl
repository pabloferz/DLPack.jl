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
    jv = DLPack.wrap(v, to_dlpack)
    dlv = DLPack.DLManagedTensor(to_dlpack(v))
    opaque_tensor = dlv.dl_tensor

    @test v.ndim == 1 == opaque_tensor.ndim
    @test opaque_tensor.dtype == DLPack.jltypes_to_dtypes()[eltype(jv)]

    if DLPack.device_type(opaque_tensor) == DLPack.kDLCPU
        jv[1] = 0  # mutate a jax's tensor
    elseif DLPack.device_type(opaque_tensor) == DLPack.kDLCUDA
        jv[1:1] .= 0  # mutate a jax's tensor
    end

    @test py"$np.all($v == $np.asarray([0.0, 2.0, 3.0])).item()"

    w = np.asarray([1 2; 3 4], dtype = np.int64)
    jw = DLPack.wrap(w, to_dlpack)
    dlw = DLPack.DLManagedTensor(to_dlpack(w))
    opaque_tensor = dlw.dl_tensor

    @test w.ndim == 2 == opaque_tensor.ndim
    @test opaque_tensor.dtype == DLPack.jltypes_to_dtypes()[eltype(jw)]

    if DLPack.device_type(opaque_tensor) == DLPack.kDLCPU
        @test jw[1, 2] == 3  # dimensions are reversed
    elseif DLPack.device_type(opaque_tensor) == DLPack.kDLCUDA
        @test all(view(dlw, 1, 2) .== 3)  # dimensions are reversed
    end

end


@testset "PythonCall" begin

    v = torch.ones((2, 4), dtype = torch.float64)
    jv = DLPack.wrap(v, torch.to_dlpack)
    dlv = DLPack.DLManagedTensor(torch.to_dlpack(v))
    opaque_tensor = dlv.dl_tensor

    @test pyconvert(Int, v.ndim) == 2 == opaque_tensor.ndim
    @test opaque_tensor.dtype == DLPack.jltypes_to_dtypes()[eltype(jv)]

    if DLPack.device_type(opaque_tensor) == DLPack.kDLCPU
        jv[5] = 0  # mutate a jax's tensor
    elseif DLPack.device_type(opaque_tensor) == DLPack.kDLCUDA
        jv[5:5] .= 0  # mutate a jax's tensor
    end

    ref = torch.tensor(((1, 1, 1, 1), (0, 1, 1, 1)), dtype = torch.float64)
    @test Bool(torch.all(v == ref).item())

end
