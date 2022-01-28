Pkg = Base.require(Base.PkgId(Base.UUID(0x44cfe95a1eb252eab672e2afdf69b78f), "Pkg"))

ENV["PYTHON"] = PythonCall.C.CTX.exe_path
Pkg.build("PyCall")
