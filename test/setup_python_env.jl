using CondaPkg

python_deps = if VERSION == v"1.6.7"
    [
        "jax<0.3",
        "libprotobuf<3.19",
        "pytorch",
        "scipy<=1.8",
    ]
else
    [
        "jax",
        "pytorch",
    ]
end

CondaPkg.add(CondaPkg.PkgREPL.parse_pkg.(python_deps))

@static if VERSION == v"1.6.7" && Sys.isapple()
    lib = joinpath(CondaPkg.envdir(), "lib/libmkl_intel_thread.1.dylib")
    run(`install_name_tool -change @rpath/libiomp5.dylib @loader_path/libiomp5.dylib $lib`)
end

# Load PythonCall after having installed all python dependencies
using PythonCall

ENV["PYTHON"] = PythonCall.C.CTX.exe_path

Pkg = Base.require(Base.PkgId(Base.UUID(0x44cfe95a1eb252eab672e2afdf69b78f), "Pkg"))
Pkg.build("PyCall")  # we rebuild PyCall to use the same python environment as PythonCall
