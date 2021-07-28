using DLPack
using PyCall
using Test


@info "Installing JAX"
#
pip = pyimport("pip")
flags = split(get(ENV, "PIPFLAGS", ""))
packages = ["jax[cpu]"]
pip.main(["install"; flags; packages])


jax = pyimport("jax")
np = pyimport("jax.numpy")
dlpack = pyimport("jax.dlpack")

@testset "DLPack.jl" begin
    v = np.asarray([1.0, 2.0, 3.0]; dtype = np.float64)
    jv = unsafe_wrap(Array, DLVector{Float64}(dlpack.to_dlpack(v)))
    jv[0] .= 0  # mutate a jax's tensor
    @test py"$np.all($v[:] == $np.asarray([0.0, 2.0, 3.0])"
end
