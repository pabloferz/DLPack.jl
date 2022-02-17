# SPDX-License-Identifier: MIT
# See LICENSE.md at https://github.com/pabloferz/DLPack.jl

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

    """
        DLManagedTensor(po::Py)

    Takes a PyCapsule holding a `DLManagedTensor` and returns the latter.
    """
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

    """
        share(A::StridedArray, ::Type{Py}, from_dlpack)

    Similar to `share(A, from_dlpack::Py)`. Use when there is a needed to
    disambiguate the return type.
    """
    function share(A::StridedArray, ::Type{Py}, from_dlpack)
        capsule = share(A)
        tensor = capsule.tensor
        tensor_ptr = pointer_from_objref(tensor)

        # Prevent `A` and `tensor` from being `gc`ed while `o` is around.
        # For certain DLPack-compatible libraries, e.g. PyTorch, the tensor is
        # captured and the `deleter` referenced from it.
        SHARES_POOL[tensor_ptr] = (capsule, A)
        tensor.deleter = PYTHONCALL_DLPACK_DELETER

        pycapsule = PythonCall.pynew(ccall(
            PyCapsule_New,
            PyPtr, (Ptr{Cvoid}, Ptr{UInt8}, Ptr{Cvoid}),
            tensor_ptr, PYCAPSULE_NAME, C_NULL
        ))

        return from_dlpack(pycapsule)
    end

end

"""
    wrap(o::Py, to_dlpack)

Takes a tensor `o::Py` and a `to_dlpack` function that generates a
`DLManagedTensor` bundled in a PyCapsule, and returns a zero-copy
`array::AbstractArray` pointing to the same data in `o`.
For tensors with row-mayor ordering the resulting array will have all
dimensions reversed.
"""
function wrap(o::Py, to_dlpack::Union{Py, Function})
    return wrap(DLManagedTensor(to_dlpack(o)), o)
end

"""
    wrap(::Type{<: AbstractArray{T, N}}, ::Type{<: MemoryLayout}, o::Py, to_dlpack)

Type-inferrable alternative to `wrap(o, to_dlpack)`.
"""
function wrap(::Type{A}, ::Type{M}, o::Py, to_dlpack) where {
    T, N, A <: AbstractArray{T, N}, M
}
    return wrap(A, M, DLManagedTensor(to_dlpack(o)), o)
end

"""
    share(A::StridedArray, from_dlpack)

Takes a Julia array and an external `from_dlpack` method that consumes PyCapsules
following the DLPack protocol. Returns a Python tensor that shares the data with `A`.
The resulting tensor will have all dimensions reversed with respect
to the Julia array.
"""
share(A::StridedArray, from_dlpack::Py) = share(A, Py, from_dlpack)
