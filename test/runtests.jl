using CUDA
using DLPack
using PyCall
using Test


# Load JAX
jax = pyimport("jax")
np = pyimport("jax.numpy")
dlpack = pyimport("jax.dlpack")


@testset "DLPack.jl" begin
    v = np.asarray([1.0, 2.0, 3.0])

    dlv = DLVector{Float32}(@pycall dlpack.to_dlpack(v)::PyObject)
    tensor = dlv.manager.dl_tensor
    
    @test tensor.ndim == 1
    @test tensor.dtype == DLPack.jltypes_to_dtypes()[Float32]

    if DLPack.device_type(dlv) == DLPack.kDLCPU
        jv = unsafe_wrap(Array, dlv)
        jv[1] = 0  # mutate a jax's tensor
    elseif DLPack.device_type(dlv) == DLPack.kDLGPU
        jv = unsafe_wrap(CuArray, dlv)
        jv[1:1] .= 0  # mutate a jax's tensor
    end

    @test py"$np.all($v[:] == $np.asarray([0.0, 2.0, 3.0])).copy().item()"
end
