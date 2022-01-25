using .PyCall

function DLManagedTensor(po::PyObject)
    if !pyisinstance(po, PyCall.@pyglobalobj(:PyCapsule_Type))
        throw(ArgumentError("PyObject must be a PyCapsule"))
    end

    # Replace the capsule destructor to prevent it from deleting the tensor.
    # We will use the `DLManagedTensor.deleter` instead
    PyCall.@pycheck ccall(
        (@pysym :PyCapsule_SetDestructor),
        Cint, (PyPtr, Ptr{Cvoid}),
        po, C_NULL
    )

    dlptr = PyCall.@pycheck ccall(
        (@pysym :PyCapsule_GetPointer),
        Ptr{DLManagedTensor}, (PyPtr, Ptr{UInt8}),
        po, ccall((@pysym :PyCapsule_GetName), Ptr{UInt8}, (PyPtr,), po)
    )

    return DLManagedTensor(dlptr)
end

DLArray(po::PyObject) = DLArray(DLManagedTensor(po))
DLArray{T, N}(po::PyObject) where {T, N} = DLArray{T, N}(DLManagedTensor(po))
