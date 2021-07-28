using DLPack
using PyCall
using Test


@info "Installing JAX"
#
pip = pyimport("pip")
flags = split(get(ENV, "PIPFLAGS", ""))
packages = ["jax[cpu]"]
pip.main(["install"; flags; packages])


@testset "DLPack.jl" begin
    # Write your tests here.
end
