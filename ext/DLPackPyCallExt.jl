# SPDX-License-Identifier: MIT
# See LICENSE.md at https://github.com/pabloferz/DLPack.jl

module DLPackPyCallExt


##  Dependencies  ##

import DLPack
@static if isdefined(Base, :get_extension)
    import PyCall
else
    import ..PyCall
end


##  Extensions  ##

const DLArray = PyCall.PyNULL()


"""
    DLManagedTensor(po::PyObject)

Takes a PyCapsule holding a `DLManagedTensor` and returns the latter.
"""
function DLPack.DLManagedTensor(po::PyCall.PyObject)
    if !PyCall.pyisinstance(po, PyCall.@pyglobalobj(:PyCapsule_Type))
        throw(ArgumentError("PyObject must be a PyCapsule"))
    end

    name = PyCall.@pycheck ccall(
        (PyCall.@pysym :PyCapsule_GetName), Ptr{Cchar}, (PyCall.PyPtr,), po
    )

    if unsafe_string(name) == "used_dltensor"
        throw(ArgumentError("PyCapsule in use, have you wrapped it already?"))
    end

    dlptr = PyCall.@pycheck ccall(
        (PyCall.@pysym :PyCapsule_GetPointer),
        Ptr{DLPack.DLManagedTensor}, (PyCall.PyPtr, Ptr{Cchar}),
        po, name
    )

    tensor = DLPack.DLManagedTensor(dlptr)

    # Replace the capsule name to "used_dltensor"
    set_name_flag = PyCall.@pycheck ccall(
        (PyCall.@pysym :PyCapsule_SetName),
        Cint, (PyCall.PyPtr, Ptr{Cchar}),
        po, DLPack.USED_PYCAPSULE_NAME
    )
    if set_name_flag != 0
        @warn("Could not mark PyCapsule as used")
    end

    # Extra precaution: Replace the capsule destructor to prevent it from deleting the
    # tensor. We will use the `DLManagedTensor.deleter` instead.
    PyCall.@pycheck ccall(
        (PyCall.@pysym :PyCapsule_SetDestructor),
        Cint, (PyCall.PyPtr, Ptr{Cvoid}),
        po, C_NULL
    )

    return tensor
end

# Docstring in src/DLPack.jl
function DLPack.from_dlpack(o::PyCall.PyObject)
    tensor = DLPack.DLManagedTensor(PyCall.@pycall o.__dlpack__()::PyCall.PyObject)
    return DLPack.unsafe_wrap(tensor, o)
end

"""
    from_dlpack(::Type{<: AbstractArray{T, N}}, ::Type{<: MemoryLayout}, o::PyObject)

Type-inferrable alternative to `from_dlpack`.
"""
function DLPack.from_dlpack(::Type{A}, ::Type{M}, o::PyCall.PyObject) where {
    T, N, A <: AbstractArray{T, N}, M
}
    tensor = DLPack.DLManagedTensor(PyCall.@pycall o.__dlpack__()::PyCall.PyObject)
    return DLPack.unsafe_wrap(A, M, tensor, o)
end

"""
    wrap(o::PyObject, to_dlpack)

Similar to `from_dlpack`, but works for python arrays that do not implement a `__dlpack__`
method. `to_dlpack` must be a function that, when applied to `o`, generates a
`DLManagedTensor` bundled into a `PyCapsule`.
"""
function DLPack.wrap(o::PyCall.PyObject, to_dlpack::Union{PyCall.PyObject, Function})
    return DLPack.unsafe_wrap(DLPack.DLManagedTensor(to_dlpack(o)), o)
end

"""
    wrap(::Type{<: AbstractArray{T, N}}, ::Type{<: MemoryLayout}, o::PyObject, to_dlpack)

Type-inferrable alternative to `wrap`.
"""
function DLPack.wrap(::Type{A}, ::Type{M}, o::PyCall.PyObject, to_dlpack) where {
    T, N, A <: AbstractArray{T, N}, M
}
    return DLPack.unsafe_wrap(A, M, DLPack.DLManagedTensor(to_dlpack(o)), o)
end

"""
    share(A::StridedArray, from_dlpack::PyObject)

Takes a Julia array and an external `from_dlpack` method that consumes PyCapsules
following the DLPack protocol. Returns a Python tensor that shares the data with `A`.
The resulting tensor will have all dimensions reversed with respect
to the Julia array.
"""
function DLPack.share(A::StridedArray, from_dlpack::PyCall.PyObject)
    capsule = DLPack.share(A)
    tensor = capsule.tensor
    tensor_ptr = pointer_from_objref(tensor)

    # Prevent `A` and `tensor` from being `gc`ed while `o` is around.
    # For certain DLPack-compatible libraries, e.g. PyTorch, the tensor is
    # captured and the `deleter` referenced from it.
    DLPack.SHARES_POOL[tensor_ptr] = (capsule, A)
    tensor.deleter = DLPack.DELETER[]

    pycapsule = PyCall.PyObject(PyCall.@pycheck ccall(
        (PyCall.@pysym :PyCapsule_New),
        PyCall.PyPtr, (Ptr{Cvoid}, Ptr{Cchar}, Ptr{Cvoid}),
        tensor_ptr, DLPack.PYCAPSULE_NAME, C_NULL
    ))

    return try
        from_dlpack(pycapsule)
    catch e
        if !(e isa KeyError && any(e.key .== (:__dlpack__, :__dlpack_device__)))
            rethrow()
        end
        dl_array = DLArray()
        ctx = DLPack.dldevice(tensor)
        dl_array.capsule = pycapsule
        dl_array.device = (Int(ctx.device_type), ctx.device_id)
        from_dlpack(dl_array)
    end
end


##  Deprecations  ##

# NOTE: replace by the following when our julia lower bound get to ≥ v"1.9".
# @deprecate(
#     DLPack.share(A::StridedArray, ::Type{PyCall.PyObject}, from_dlpack),
#     DLPack.share(A, PyCall.pyfunction(from_dlpack, PyCall.PyObject)),
#     false
# )
function DLPack.share(A::StridedArray, ::Type{PyCall.PyObject}, from_dlpack)
    Base.depwarn("""
        `DLPack.share`(A, ::Type{PyObject}), from_dlpack) is deprecated, use
        `DLPack.share`(A, from_dlpack) instead. If `from_dlpack` is a julia function,
        use `pyfunction` to wrap it.
        """,
        :share
    )
    DLPack.share(A, PyCall.pyfunction(from_dlpack, PyCall.PyObject))
end


##  Extension initialization  ##

function __init__()
    copy!(DLArray,
        PyCall.@pydef_object mutable struct DLArray
            capsule = nothing
            device = nothing
            __dlpack__(self; stream = nothing) = self."capsule"
            __dlpack_device__(self) = self."device"
        end
    )
end


end  # module DLPackPyCallExt
