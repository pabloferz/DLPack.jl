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
    dlv = DLPack.DLManagedTensor(to_dlpack(v))
    opaque_tensor = dlv.dl_tensor

    @test v.ndim == 1 == opaque_tensor.ndim
    @test opaque_tensor.dtype == DLPack.jltypes_to_dtypes()[Float32]

    if DLPack.device_type(opaque_tensor) == DLPack.kDLCPU
        # dlv[1] = 0  # mutate a jax's tensor
    elseif DLPack.device_type(opaque_tensor) == DLPack.kDLCUDA
        # dlv[1:1] .= 0  # mutate a jax's tensor
    end

    # @test py"$np.all($v[:] == $np.asarray([0.0, 2.0, 3.0])).item()"

    w = np.asarray([1 2; 3 4], dtype = np.int64)
    dlw = DLPack.DLManagedTensor(to_dlpack(w))
    opaque_tensor = dlw.dl_tensor

    @test w.ndim == 2 == opaque_tensor.ndim
    @test opaque_tensor.dtype == DLPack.jltypes_to_dtypes()[Int64]

    if DLPack.device_type(opaque_tensor) == DLPack.kDLCPU
        # @test dlw[2, 1] == 3
    elseif DLPack.device_type(opaque_tensor) == DLPack.kDLCUDA
        # @test all(view(dlw, 2, 1) .== 3)
    end

end


@testset "PythonCall" begin

    v = torch.ones((2, 4), dtype = torch.float64)
    dlv = DLPack.DLManagedTensor(torch.to_dlpack(v))
    opaque_tensor = dlv.dl_tensor

    @test pyconvert(Int, v.ndim) == 2 == opaque_tensor.ndim
    @test opaque_tensor.dtype == DLPack.jltypes_to_dtypes()[Float64]

    if DLPack.device_type(opaque_tensor) == DLPack.kDLCPU
        # dlv[2] = 0  # mutate a jax's tensor
    elseif DLPack.device_type(opaque_tensor) == DLPack.kDLCUDA
        # dlv[2:2] .= 0  # mutate a jax's tensor
    end

    # ref = torch.tensor(((1, 1, 1, 1), (0, 1, 1, 1)), dtype = torch.float64)
    # @test Bool(torch.all(v == ref).item())

end
