using CondaPkg

python_deps = if VERSION == v"1.6.7"
    [
        "jax<0.3",
        "libprotobuf<3.19",
        "numpy<1.25",
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
