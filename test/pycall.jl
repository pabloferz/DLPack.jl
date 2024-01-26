@testitem "PyCall" begin

    using CUDA
    using PyCall


    const jax = pyimport("jax")
    const jnp = pyimport("jax.numpy")
    const dlpack = pyimport("jax.dlpack")

    jax.config.update("jax_enable_x64", true)


    @testset "wrap" begin
        to_dlpack = o -> @pycall dlpack.to_dlpack(o)::PyObject

        v = jnp.asarray([1.0, 2.0, 3.0], dtype = jnp.float32)
        jv = DLPack.wrap(v, to_dlpack)
        dlv = DLPack.DLManagedTensor(to_dlpack(v))
        opaque_tensor = dlv.dl_tensor

        @test v.ndim == 1 == opaque_tensor.ndim
        @test opaque_tensor.dtype == DLPack.jltypes_to_dtypes()[eltype(jv)]

        if DLPack.device_type(opaque_tensor) == DLPack.kDLCPU
            jv[1] = 0  # mutate a jax tensor
            @inferred DLPack.wrap(Vector{Float32}, ColMajor, v, to_dlpack)
        elseif DLPack.device_type(opaque_tensor) == DLPack.kDLCUDA
            jv[1:1] .= 0  # mutate a jax tensor
        end

        @test py"$jnp.all($v == $jnp.asarray([0.0, 2.0, 3.0])).item()"

        w = jnp.asarray([1 2; 3 4], dtype = jnp.int64)
        jw = DLPack.wrap(w, to_dlpack)
        dlw = DLPack.DLManagedTensor(to_dlpack(w))
        opaque_tensor = dlw.dl_tensor

        @test w.ndim == 2 == opaque_tensor.ndim
        @test opaque_tensor.dtype == DLPack.jltypes_to_dtypes()[eltype(jw)]

        if DLPack.device_type(opaque_tensor) == DLPack.kDLCPU
            @test jw[1, 2] == 3  # dimensions are reversed
            @inferred DLPack.wrap(Matrix{Int64}, RowMajor, w, to_dlpack)
        elseif DLPack.device_type(opaque_tensor) == DLPack.kDLCUDA
            @test all(view(jw, 1, 2) .== 3)  # dimensions are reversed
        end
    end

    @testset "share" begin
        v = UInt[1, 2, 3]
        pv = DLPack.share(v, dlpack.from_dlpack)

        @test ndims(v) == 1 == pv.ndim
        @test pv.dtype == jnp.dtype("uint64")

        v[1] = 0
        @test py"($pv[0] == 0).item()"

        w = rand(2, 2)
        pw = DLPack.share(w, dlpack.from_dlpack)

        @test ndims(w) == 2 == pw.ndim
        @test pw.dtype == jnp.dtype("float64")
        @test w[1, 2] == py"($pw[1, 0]).item()"  # dimensions are reversed
    end

end
