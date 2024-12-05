@testitem "Aqua" begin
    import Aqua

    Aqua.test_all(DLPack; deps_compat = (; check_extras = false))
    Aqua.test_deps_compat(Aqua.aspkgid(DLPack), "extras"; ignore = [:CondaPkg, :Test])
end
