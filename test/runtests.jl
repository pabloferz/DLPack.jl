if lowercase(get(ENV, "CI", "false")) == "true"
    include("install_dependencies.jl")
end


using CUDA
using DLPack
using PyCall
using Test


@info "Installing JAX"
run(PyCall.python_cmd(`-m pip install --upgrade jax\[cpu\]`))


# Load JAX
jax = pyimport("jax")
np = pyimport("jax.numpy")
dlpack = pyimport("jax.dlpack")


@testset "DLPack.jl" begin
    v = np.asarray([1.0, 2.0, 3.0])

    dlv = DLVector{Float32}(@pycall dlpack.to_dlpack(v)::PyObject)
    tensv = dlv.manager.dl_tensor

    @test tensv.ndim == 1
    @test tensv.dtype == DLPack.jltypes_to_dtypes()[Float32]

    if DLPack.device_type(dlv) == DLPack.kDLCPU
        jv = unsafe_wrap(Array, dlv)
        jv[1] = 0  # mutate a jax's tensor
    elseif DLPack.device_type(dlv) == DLPack.kDLCUDA
        jv = unsafe_wrap(CuArray, dlv)
        jv[1:1] .= 0  # mutate a jax's tensor
    end

    @test py"$np.all($v[:] == $np.asarray([0.0, 2.0, 3.0])).item()"

    w = np.asarray([1.0 2.0; 3.0 4.0])

    dlw = DLArray(@pycall dlpack.to_dlpack(w)::PyObject)
    tensw = dlw.manager.dl_tensor

    @test tensw.ndim == 2
    @test tensw.dtype == DLPack.jltypes_to_dtypes()[Float32]

    if DLPack.device_type(dlw) == DLPack.kDLCPU
        jw = unsafe_wrap(Array, dlw)
        @test jw[2, 1] == 2.0
    elseif DLPack.device_type(dlw) == DLPack.kDLCUDA
        jw = unsafe_wrap(CuArray, dlw)
        @test all(view(jw, 2, 1) .== 2.0)
    end
end
