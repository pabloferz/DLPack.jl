Pkg = Base.require(Base.PkgId(Base.UUID(0x44cfe95a1eb252eab672e2afdf69b78f), "Pkg"))
using CondaPkg

Pkg.pkg"conda add libprotobuf<3.19 scipy<=1.8 jax<=0.3 pytorch"
