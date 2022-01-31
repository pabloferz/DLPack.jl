let

    global DLManagedTensor

    Libdl = Base.require(
        Base.PkgId(Base.UUID("8f399da3-3557-5675-b5ff-fb832c97cbdb"), "Libdl")
    )

    lib_ptr = PythonCall.C.CTX.lib_ptr
    PyPtr = PythonCall.C.PyPtr

    # PythonCall doesn't expose these
    PyCapsule_Type_Ptr = PyPtr(Libdl.dlsym(lib_ptr, :PyCapsule_Type))
    PyCapsule_GetName_Ptr = Libdl.dlsym(lib_ptr, :PyCapsule_GetName)
    PyCapsule_SetName_Ptr = Libdl.dlsym(lib_ptr, :PyCapsule_SetName)
    PyCapsule_GetPointer_Ptr = Libdl.dlsym(lib_ptr, :PyCapsule_GetPointer)
    PyCapsule_SetDestructor_Ptr = Libdl.dlsym(lib_ptr, :PyCapsule_SetDestructor)

    function DLManagedTensor(po::PythonCall.Py)
        ptr = PythonCall.getptr(po)

        if PythonCall.C.PyObject_IsInstance(ptr, PyCapsule_Type_Ptr) != 1
            throw(ArgumentError("PyObject must be a PyCapsule"))
        end

        name = ccall(PyCapsule_GetName_Ptr, Ptr{UInt8}, (PyPtr,), ptr)

        if unsafe_string(name) == "used_dltensor"
            throw(ArgumentError("PyCapsule in use, have you wrapped it already?"))
        end

        dlptr = ccall(
            PyCapsule_GetPointer_Ptr,
            Ptr{DLManagedTensor}, (PyPtr, Ptr{UInt8}),
            ptr, name
        )

        tensor = DLManagedTensor(dlptr)

        # Replace the capsule name to "used_dltensor"
        set_name_flag = ccall(
            PyCapsule_SetName_Ptr, Cint, (PyPtr, Ptr{UInt8}), ptr, USED_PYCAPSULE_NAME
        )
        if set_name_flag != 0
            @warn("Could not mark PyCapsule as used")
        end

        # Extra precaution: Replace the capsule destructor to prevent it from deleting the
        # tensor. We will use the `DLManagedTensor.deleter` instead.
        ccall(PyCapsule_SetDestructor_Ptr, Cint, (PyPtr, Ptr{Cvoid}), ptr, C_NULL)

        return tensor
    end

end

DLArray(o::PythonCall.Py, to_dlpack) = DLArray(DLManagedTensor(to_dlpack(o)), o)
#
function DLArray{T, N}(::Type{A}, ::Type{M}, o::PythonCall.Py, to_dlpack) where {T, N, A, M}
    return DLArray{T, N}(A, M, DLManagedTensor(to_dlpack(o)), o)
end
