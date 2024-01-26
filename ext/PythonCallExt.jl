# SPDX-License-Identifier: MIT
# See LICENSE.md at https://github.com/pabloferz/DLPack.jl

module PythonCallExt


##  Dependencies  ##

import DLPack
@static if isdefined(Base, :get_extension)
    import PythonCall
else
    import ..PythonCall
end


##  Extensions  ##

const CPython = PythonCall.C


"""
    DLManagedTensor(po::Py)

Takes a PyCapsule holding a `DLManagedTensor` and returns the latter.
"""
function DLPack.DLManagedTensor(po::PythonCall.Py)
    ptr = PythonCall.getptr(po)

    if CPython.PyObject_IsInstance(ptr, CPython.POINTERS.PyCapsule_Type) != 1
        throw(ArgumentError("PyObject must be a PyCapsule"))
    end

    name = CPython.PyCapsule_GetName(ptr)

    if unsafe_string(name) == "used_dltensor"
        throw(ArgumentError("PyCapsule in use, have you wrapped it already?"))
    end

    dlptr = Ptr{DLPack.DLManagedTensor}(CPython.PyCapsule_GetPointer(ptr, name))
    tensor = DLPack.DLManagedTensor(dlptr)

    # Replace the capsule name to "used_dltensor"
    set_name_flag = CPython.PyCapsule_SetName(ptr, DLPack.USED_PYCAPSULE_NAME)

    if set_name_flag != 0
        @warn("Could not mark PyCapsule as used")
    end

    # Extra precaution: Replace the capsule destructor to prevent it from deleting the
    # tensor. We will use the `DLManagedTensor.deleter` instead.
    CPython.PyCapsule_SetDestructor(ptr, C_NULL)

    return tensor
end

"""
    wrap(o::Py, to_dlpack)

Takes a tensor `o::Py` and a `to_dlpack` function that generates a
`DLManagedTensor` bundled in a PyCapsule, and returns a zero-copy
`array::AbstractArray` pointing to the same data in `o`.
For tensors with row-major ordering the resulting array will have all
dimensions reversed.
"""
function DLPack.wrap(o::PythonCall.Py, to_dlpack::Union{PythonCall.Py, Function})
    return DLPack.unsafe_wrap(DLPack.DLManagedTensor(to_dlpack(o)), o)
end

"""
    wrap(::Type{<: AbstractArray{T, N}}, ::Type{<: MemoryLayout}, o::Py, to_dlpack)

Type-inferrable alternative to `wrap(o, to_dlpack)`.
"""
function DLPack.wrap(::Type{A}, ::Type{M}, o::PythonCall.Py, to_dlpack) where {
    T, N, A <: AbstractArray{T, N}, M
}
    return DLPack.unsafe_wrap(A, M, DLPack.DLManagedTensor(to_dlpack(o)), o)
end

"""
    share(A::StridedArray, from_dlpack::Py)

Takes a Julia array and an external `from_dlpack` method that consumes PyCapsules
following the DLPack protocol. Returns a Python tensor that shares the data with `A`.
The resulting tensor will have all dimensions reversed with respect
to the Julia array.
"""
DLPack.share(A::StridedArray, from_dlpack::PythonCall.Py) = DLPack.share(A, PythonCall.Py, from_dlpack)

"""
    share(A::StridedArray, ::Type{Py}, from_dlpack)

Similar to `share(A, from_dlpack::Py)`. Use when there is a need to
disambiguate the return type.
"""
function DLPack.share(A::StridedArray, ::Type{PythonCall.Py}, from_dlpack)
    capsule = DLPack.share(A)
    tensor = capsule.tensor
    tensor_ptr = pointer_from_objref(tensor)

    # Prevent `A` and `tensor` from being `gc`ed while `o` is around.
    # For certain DLPack-compatible libraries, e.g. PyTorch, the tensor is
    # captured and the `deleter` referenced from it.
    DLPack.SHARES_POOL[tensor_ptr] = (capsule, A)
    tensor.deleter = DLPack.DELETER[]

    pycapsule = PythonCall.pynew(
        CPython.PyCapsule_New(tensor_ptr, DLPack.PYCAPSULE_NAME, C_NULL)
    )

    return from_dlpack(pycapsule)
end


end  # module PythonCallExt
