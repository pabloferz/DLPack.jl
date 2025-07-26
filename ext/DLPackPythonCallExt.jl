# SPDX-License-Identifier: MIT
# See LICENSE.md at https://github.com/pabloferz/DLPack.jl

module DLPackPythonCallExt


##  Dependencies  ##

import DLPack
@static if isdefined(Base, :get_extension)
    import PythonCall
else
    import ..PythonCall
end


##  Extensions  ##

const CPython = PythonCall.C
const DLArray = PythonCall.pynew()
const PyTypes = Union{PythonCall.Py, PythonCall.PyArray, PythonCall.PyIterable}


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

# Docstring in src/DLPack.jl
function DLPack.from_dlpack(o::PyTypes)
    py = PythonCall.Py(o)
    tensor = DLPack.DLManagedTensor(py.__dlpack__())
    return DLPack.unsafe_wrap(tensor, py)
end

"""
    from_dlpack(::Type{<: AbstractArray{T, N}}, ::Type{<: MemoryLayout}, o::Py)

Type-inferrable alternative to `from_dlpack`.
"""
function DLPack.from_dlpack(::Type{A}, ::Type{M}, o::PyTypes) where {
    T, N, A <: AbstractArray{T, N}, M
}
    py = PythonCall.Py(o)
    tensor = DLPack.DLManagedTensor(py.__dlpack__())
    return DLPack.unsafe_wrap(A, M, tensor, py)
end

"""
    wrap(o::Py, to_dlpack)

Similar to `from_dlpack`, but works for python arrays that do not implement a `__dlpack__`
method. `to_dlpack` must be a function that, when applied to `o`, generates a
`DLManagedTensor` bundled into a `PyCapsule`.
"""
function DLPack.wrap(o::PyTypes, to_dlpack::Union{PythonCall.Py, Function})
    py = PythonCall.Py(o)
    return DLPack.unsafe_wrap(DLPack.DLManagedTensor(to_dlpack(py)), py)
end

"""
    wrap(::Type{<: AbstractArray{T, N}}, ::Type{<: MemoryLayout}, o::Py, to_dlpack)

Type-inferrable alternative to `wrap`.
"""
function DLPack.wrap(::Type{A}, ::Type{M}, o::PyTypes, to_dlpack) where {
    T, N, A <: AbstractArray{T, N}, M
}
    py = PythonCall.Py(o)
    return DLPack.unsafe_wrap(A, M, DLPack.DLManagedTensor(to_dlpack(py)), py)
end

"""
    share(A::StridedArray, from_dlpack::Py)

Takes a Julia array and an external `from_dlpack` method that consumes PyCapsules
following the DLPack protocol. Returns a Python tensor that shares the data with `A`.
The resulting tensor will have all dimensions reversed with respect
to the Julia array.
"""
function DLPack.share(A::StridedArray, from_dlpack::PythonCall.Py)
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

    return try
        from_dlpack(pycapsule)
    catch e
        if !(
            PythonCall.pyisinstance(e, PythonCall.pybuiltins.AttributeError) &&
            any(contains.(string(PythonCall.Py(e)), ("__dlpack__", "__dlpack_device__")))
        )
            rethrow()
        end
        ctx = DLPack.dldevice(tensor)
        device = (Int(ctx.device_type), ctx.device_id)
        from_dlpack(DLArray(pycapsule, device))
    end
end


##  Deprecations  ##

# NOTE: replace by the following when our julia lower bound get to ≥ v"1.9".
# @deprecate(
#     DLPack.share(A::StridedArray, ::Type{PythonCall.Py}, from_dlpack),
#     DLPack.share(A, PythonCall.pyfunc(from_dlpack)),
#     #= export_old =# false
# )
function DLPack.share(A::StridedArray, ::Type{PythonCall.Py}, from_dlpack)
    Base.depwarn("""
        `DLPack.share`(A, ::Type{Py}), from_dlpack) is deprecated, use
        `DLPack.share`(A, from_dlpack) instead. If `from_dlpack` is a julia function,
        use `pyfunc` to wrap it.
        """,
        :share
    )
    DLPack.share(A, PythonCall.pyfunc(from_dlpack))
end


##  Extension initialization  ##

function __init__()
    PythonCall.pycopy!(DLArray,
        PythonCall.pytype("DLArray", (), [
            "__module__" => "__main__",

            PythonCall.pyfunc(
                name = "__init__",
                (self, capsule, device) -> begin
                    self.capsule = capsule
                    self.device = device
                    nothing
                end,
            ),

            PythonCall.pyfunc(
                name = "__dlpack__",
                (self; stream = nothing) -> self.capsule,
            ),

            PythonCall.pyfunc(
                name = "__dlpack_device__",
                (self) -> self.device,
            )
        ])
    )
end


end  # module DLPackPythonCallExt
