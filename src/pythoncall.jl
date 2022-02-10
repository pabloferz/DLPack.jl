using .PythonCall


# This will be used to release the `DLManagedTensor`s and associated array.
const PYTHONCALL_DLPACK_DELETER = @cfunction(release, Cvoid, (Ptr{Cvoid},))


let

    global DLManagedTensor
    global share

    Libdl = Base.require(
        Base.PkgId(Base.UUID("8f399da3-3557-5675-b5ff-fb832c97cbdb"), "Libdl")
    )

    PyPtr = PythonCall.C.PyPtr
    lib_ptr = PythonCall.C.CTX.lib_ptr

    # PythonCall doesn't expose these
    PyCapsule_Type = PyPtr(Libdl.dlsym(lib_ptr, :PyCapsule_Type))
    PyCapsule_New = Libdl.dlsym(lib_ptr, :PyCapsule_New)
    PyCapsule_GetName = Libdl.dlsym(lib_ptr, :PyCapsule_GetName)
    PyCapsule_SetName = Libdl.dlsym(lib_ptr, :PyCapsule_SetName)
    PyCapsule_GetPointer = Libdl.dlsym(lib_ptr, :PyCapsule_GetPointer)
    PyCapsule_SetDestructor = Libdl.dlsym(lib_ptr, :PyCapsule_SetDestructor)

    function DLManagedTensor(po::Py)
        ptr = PythonCall.getptr(po)

        if PythonCall.C.PyObject_IsInstance(ptr, PyCapsule_Type) != 1
            throw(ArgumentError("PyObject must be a PyCapsule"))
        end

        name = ccall(PyCapsule_GetName, Ptr{UInt8}, (PyPtr,), ptr)

        if unsafe_string(name) == "used_dltensor"
            throw(ArgumentError("PyCapsule in use, have you wrapped it already?"))
        end

        dlptr = ccall(
            PyCapsule_GetPointer,
            Ptr{DLManagedTensor}, (PyPtr, Ptr{UInt8}),
            ptr, name
        )

        tensor = DLManagedTensor(dlptr)

        # Replace the capsule name to "used_dltensor"
        set_name_flag = ccall(
            PyCapsule_SetName, Cint, (PyPtr, Ptr{UInt8}), ptr, USED_PYCAPSULE_NAME
        )
        if set_name_flag != 0
            @warn("Could not mark PyCapsule as used")
        end

        # Extra precaution: Replace the capsule destructor to prevent it from deleting the
        # tensor. We will use the `DLManagedTensor.deleter` instead.
        ccall(PyCapsule_SetDestructor, Cint, (PyPtr, Ptr{Cvoid}), ptr, C_NULL)

        return tensor
    end

    function share(A::StridedArray, ::Type{Py}, from_dlpack)
        capsule = share(A)
        tensor = capsule.tensor
        tensor_ptr = pointer_from_objref(tensor)

        # Prevent `A` and `tensor` from being `gc`ed while `o` is around.
        # For certain DLPack-compatible libraries, e.g. PyTorch, the tensor is
        # captured and the `deleter` referenced from it.
        DLPACK_POOL[tensor_ptr] = (capsule, A)
        tensor.deleter = PYTHONCALL_DLPACK_DELETER

        pycapsule = PythonCall.pynew(ccall(
            PyCapsule_New,
            PyPtr, (Ptr{Cvoid}, Ptr{UInt8}, Ptr{Cvoid}),
            tensor_ptr, PYCAPSULE_NAME, C_NULL
        ))

        return from_dlpack(pycapsule)
    end

end

share(A::StridedArray, from_dlpack::Py) = share(A, Py, from_dlpack)

function DLArray(o::Py, to_dlpack::Union{Py, Function})
    return DLArray(DLManagedTensor(to_dlpack(o)), o)
end
#
function DLArray{T, N}(::Type{A}, ::Type{M}, o::Py, to_dlpack) where {T, N, A, M}
    return DLArray{T, N}(A, M, DLManagedTensor(to_dlpack(o)), o)
end
