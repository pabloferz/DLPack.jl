@testitem "PythonCall" begin

    using CUDA
    using PythonCall


    const torch = pyimport("torch")
    const np = pyimport("numpy")


    @testset "wrap" begin
        v = torch.ones((2, 4), dtype = torch.float64)
        follows_dlpack_spec = hasproperty(v, :__dlpack__)
        jv = follows_dlpack_spec ? DLPack.from_dlpack(v) : DLPack.wrap(v, torch.to_dlpack)
        dlv = DLPack.DLManagedTensor(torch.to_dlpack(v))
        opaque_tensor = dlv.dl_tensor

        @test pyconvert(Int, v.ndim) == 2 == opaque_tensor.ndim
        @test opaque_tensor.dtype == DLPack.jltypes_to_dtypes()[eltype(jv)]

        if DLPack.device_type(opaque_tensor) == DLPack.kDLCPU
            jv[5] = 0  # mutate a torch tensor
        elseif DLPack.device_type(opaque_tensor) == DLPack.kDLCUDA
            jv[5:5] .= 0  # mutate a torch tensor
        end

        ref = torch.tensor(((1, 1, 1, 1), (0, 1, 1, 1)), dtype = torch.float64)
        @test Bool(torch.all(v == ref).item())
    end

    @testset "share" begin
        np_from_dlpack = if haskey(np.__dict__, "_from_dlpack")
            np._from_dlpack
        else
            np.from_dlpack
        end

        v = ComplexF32[1, 2, 3]
        npv = DLPack.share(v, np_from_dlpack)
        tv = DLPack.share(v, torch.from_dlpack)

        @test all(Bool ∘ ==(1), (ndims(v), npv.ndim, tv.ndim))
        @test Bool(npv.dtype == np.complex64)
        @test Bool(tv.dtype == torch.complex64)

        v[1] = 0
        @test all(Bool ∘ ==(0), (npv[0], tv[0]))

        w = rand(2, 2)
        npw = DLPack.share(w, np_from_dlpack)
        tw = DLPack.share(w, torch.from_dlpack)

        @test all(Bool ∘ ==(2), (ndims(w), npw.ndim, tw.ndim))
        @test Bool(npw.dtype == np.float64)
        @test Bool(tw.dtype == torch.float64)
        @test all(Bool ∘ ==(w[1, 2]), (npw[1, 0], tw[1, 0]))
    end

end
