let

    global DLManagedTensor

    Libdl = Base.require(
        Base.PkgId(Base.UUID("8f399da3-3557-5675-b5ff-fb832c97cbdb"), "Libdl")
    )

    lib_ptr = PythonCall.C.CTX.lib_ptr
    PyPtr = PythonCall.C.PyPtr

    # PythonCall doesn't exposes these
    PyCapsule_Type = PyPtr(Libdl.dlsym(lib_ptr, :PyCapsule_Type))
    PyCapsule_GetName = Libdl.dlsym(lib_ptr, :PyCapsule_GetName)
    PyCapsule_GetPointer = Libdl.dlsym(lib_ptr, :PyCapsule_GetPointer)
    PyCapsule_SetDestructor = Libdl.dlsym(lib_ptr, :PyCapsule_SetDestructor)

    function DLManagedTensor(po::PythonCall.Py)
        ptr = PythonCall.getptr(po)

        if PythonCall.C.PyObject_IsInstance(ptr, PyCapsule_Type) != 1
            throw(ArgumentError("PyObject must be a PyCapsule"))
        end

        # Replace the capsule destructor to prevent it from deleting the tensor.
        # We will use the `DLManagedTensor.deleter` instead
        ccall(PyCapsule_SetDestructor, Cint, (PyPtr, Ptr{Cvoid}), ptr, C_NULL)

        dlptr = ccall(
            PyCapsule_GetPointer,
            Ptr{DLManagedTensor}, (PyPtr, Ptr{UInt8}),
            ptr, ccall(PyCapsule_GetName, Ptr{UInt8}, (PyPtr,), ptr)
        )

        return DLManagedTensor(dlptr)
    end

end

DLArray(po::PythonCall.Py) = DLArray(DLManagedTensor(po))
DLArray{T, N}(po::PythonCall.Py) where {T, N} = DLArray{T, N}(DLManagedTensor(po))
