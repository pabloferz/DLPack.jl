using CUDA
using DLPack
using PyCall
using Test


@info "Installing JAX"
#
packages = "jax[cpu]"
run(PyCall.python_cmd(`-m pip install $packages`))


# Load JAX
jax = pyimport("jax")
np = pyimport("jax.numpy")
dlpack = pyimport("jax.dlpack")


@testset "DLPack.jl" begin
    v = np.asarray([1.0, 2.0, 3.0])

    dlv = DLVector{Float32}(@pycall dlpack.to_dlpack(v)::PyObject)
    if DLPack.device_type(dlv) == DLPack.kDLCPU
        jv = unsafe_wrap(Array, dlv)
        jv[1] = 0  # mutate a jax's tensor
    elseif DLPack.device_type(dlv) == DLPack.kDLGPU
        jv = unsafe_wrap(CuArray, dlv)
        jv[1:1] .= 0  # mutate a jax's tensor
    end

    @test py"$np.all($v[:] == $np.asarray([0.0, 2.0, 3.0])).copy().item()"
end
