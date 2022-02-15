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

    @testset "wrap" begin
        to_dlpack = o -> @pycall dlpack.to_dlpack(o)::PyObject

        v = np.asarray([1.0, 2.0, 3.0], dtype = np.float32)
        jv = DLPack.wrap(v, to_dlpack)
        dlv = DLPack.DLManagedTensor(to_dlpack(v))
        opaque_tensor = dlv.dl_tensor

        @test v.ndim == 1 == opaque_tensor.ndim
        @test opaque_tensor.dtype == DLPack.jltypes_to_dtypes()[eltype(jv)]

        if DLPack.device_type(opaque_tensor) == DLPack.kDLCPU
            jv[1] = 0  # mutate a jax's tensor
            @inferred DLPack.wrap(Vector{Float32}, ColMajor, v, to_dlpack)
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
            @inferred DLPack.wrap(Matrix{Int64}, RowMajor, w, to_dlpack)
        elseif DLPack.device_type(opaque_tensor) == DLPack.kDLCUDA
            @test all(view(dlw, 1, 2) .== 3)  # dimensions are reversed
        end
    end

    @testset "share" begin
        v = UInt[1, 2, 3]
        pv = DLPack.share(v, dlpack.from_dlpack)

        @test ndims(v) == 1 == pv.ndim
        @test pv.dtype == np.dtype("uint64")

        v[1] = 0
        @test py"($pv[0] == 0).item()"

        w = rand(2, 2)
        pw = DLPack.share(w, dlpack.from_dlpack)

        @test ndims(w) == 2 == pw.ndim
        @test pw.dtype == np.dtype("float64")
        @test w[1, 2] == py"($pw[1, 0]).item()"  # dimensions are reversed
    end

end


@testset "PythonCall" begin

    @testset "wrap" begin
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

    @testset "share" begin
        v = ComplexF32[1, 2, 3]
        pv = DLPack.share(v, torch.from_dlpack)

        @test Bool(ndims(v) == 1 == pv.ndim)
        @test Bool(pv.dtype == torch.complex64)

        v[1] = 0
        @test Bool(pv[0] == 0)

        w = rand(2, 2)
        pw = DLPack.share(w, torch.from_dlpack)

        @test Bool(ndims(w) == 2 == pw.ndim)
        @test Bool(pw.dtype == torch.float64)
        @test Bool(w[1, 2] == pw[1, 0])
    end

end
