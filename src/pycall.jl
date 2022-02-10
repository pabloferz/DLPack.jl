using .PyCall


# We define a noop deleter to pass to new `DLManagedTensor`s exported to python libraries
# as some of them (e.g. PyTorch) do not handle the case when the finalizer is `C_NULL`.
const PYCALL_DLPACK_DELETER = @cfunction(release, Cvoid, (Ptr{Cvoid},))


function DLManagedTensor(po::PyObject)
    if !pyisinstance(po, PyCall.@pyglobalobj(:PyCapsule_Type))
        throw(ArgumentError("PyObject must be a PyCapsule"))
    end

    name = PyCall.@pycheck ccall((@pysym :PyCapsule_GetName), Ptr{UInt8}, (PyPtr,), po)

    if unsafe_string(name) == "used_dltensor"
        throw(ArgumentError("PyCapsule in use, have you wrapped it already?"))
    end

    dlptr = PyCall.@pycheck ccall(
        (@pysym :PyCapsule_GetPointer),
        Ptr{DLManagedTensor}, (PyPtr, Ptr{UInt8}),
        po, name
    )

    tensor = DLManagedTensor(dlptr)

    # Replace the capsule name to "used_dltensor"
    set_name_flag = PyCall.@pycheck ccall(
        (@pysym :PyCapsule_SetName), Cint, (PyPtr, Ptr{UInt8}), po, USED_PYCAPSULE_NAME
    )
    if set_name_flag != 0
        @warn("Could not mark PyCapsule as used")
    end

    # Extra precaution: Replace the capsule destructor to prevent it from deleting the
    # tensor. We will use the `DLManagedTensor.deleter` instead.
    PyCall.@pycheck ccall(
        (@pysym :PyCapsule_SetDestructor), Cint, (PyPtr, Ptr{Cvoid}), po, C_NULL
    )

    return tensor
end

function DLArray(o::PyObject, to_dlpack::Union{PyObject, Function})
    return DLArray(DLManagedTensor(to_dlpack(o)), o)
end
#
function DLArray{T, N}(::Type{A}, ::Type{M}, o::PyObject, to_dlpack) where {T, N, A, M}
    return DLArray{T, N}(A, M, DLManagedTensor(to_dlpack(o)), o)
end

function share(A::StridedArray, from_dlpack::Union{PyObject, Function})
    capsule = share(A)
    tensor = capsule.tensor
    tensor_ptr = pointer_from_objref(tensor)

    # Prevent `A` and `tensor` from being `gc`ed while `o` is around.
    # For certain DLPack-compatible libraries, e.g. PyTorch, the tensor is
    # captured and the `deleter` referenced from it.
    DLPACK_POOL[tensor_ptr] = (capsule, A)
    tensor.deleter = PYCALL_DLPACK_DELETER

    pycapsule = PyObject(PyCall.@pycheck ccall(
        (@pysym :PyCapsule_New),
        PyPtr, (Ptr{Cvoid}, Ptr{UInt8}, Ptr{Cvoid}),
        tensor_ptr, PYCAPSULE_NAME, C_NULL
    ))

    return from_dlpack(pycapsule)
end
