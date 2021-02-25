module DLPack


using CUDA  # For exporting to CuArrays
using PyCall


export DLArray, DLMatrix, DLVector


@enum DLDeviceType::Cint begin
    kDLCPU = 1
    kDLGPU = 2
    kDLCPUPinned = 3
    kDLOpenCL = 4
    kDLVulkan = 7
    kDLMetal = 8
    kDLVPI = 9
    kDLROCM = 10
    kDLExtDev = 12
end

struct DLContext
    device_type::Cint
    device_id::Cint
end

@enum DLDataTypeCode::Cuint begin
    kDLInt = 0
    kDLUInt = 1
    kDLFloat = 2
    kDLBfloat = 4
end

struct DLDataType
    code::Cuchar
    bits::Cuchar
    lanes::Cushort
end

Base.convert(::Type{T}, code::DLDataTypeCode) where {T <: Integer} = T(code)

jltypes_to_dtypes() = Dict(
    Int8 => DLDataType(kDLInt, 8, 1),
    Int16 => DLDataType(kDLInt, 16, 1),
    Int32 => DLDataType(kDLInt, 32, 1),
    Int64 => DLDataType(kDLInt, 64, 1),
    UInt8 => DLDataType(kDLUInt, 8, 1),
    UInt16 => DLDataType(kDLUInt, 16, 1),
    UInt32 => DLDataType(kDLUInt, 32, 1),
    UInt64 => DLDataType(kDLUInt, 64, 1),
    Float16 => DLDataType(kDLFloat, 16, 1),
    Float32 => DLDataType(kDLFloat, 32, 1),
    Float64 => DLDataType(kDLFloat, 64, 1)
)

struct DLTensor
    data::Ptr{Cvoid}
    ctx::DLContext
    ndim::Cint
    dtype::DLDataType
    shape::Ptr{Clonglong}
    strides::Ptr{Clonglong}
    byte_offset::Culonglong
end

# Defined as mutable since we need a finalizer that calls `deleter`
# to destroy its original enclosing context `manager_ctx`
mutable struct DLManagedTensor
    dl_tensor::DLTensor
    manager_ctx::Ptr{Cvoid}
    deleter::Ptr{Cvoid}
end

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
    manager = unsafe_load(dlptr)

    if manager.deleter != C_NULL
        delete = manager -> ccall(manager.deleter, Cvoid, (Ptr{Cvoid},), Ref(manager))
        finalizer(delete, manager)
    end

    return manager
end

struct DLArray{T, N}
    manager::DLManagedTensor

    function DLArray{DLDataType, N}(po::PyObject) where {T, N}
        manager = DLManagedTensor(po)

        if N != (n = manager.dl_tensor.ndim)
            throw(ArgumentError("Dimensionality mismatch, object ndims is $n"))
        end

        return new(manager)
    end

    function DLArray{T, N}(po::PyObject) where {T, N}
        manager = DLManagedTensor(po)

        if N != (n = manager.dl_tensor.ndim)
            throw(ArgumentError("Dimensionality mismatch, object ndims is $n"))
        elseif jltypes_to_dtypes()[T] != (D = manager.dl_tensor.dtype)
            throw(ArgumentError("Type mismatch, object dtype is $D"))
        end

        return new(manager)
    end
end

const DLVector{T} = DLArray{T, 1}
const DLMatrix{T} = DLArray{T, 2}

###########
#  Utils  #
###########

device_type(ctx::DLContext) = DLDeviceType(ctx.device_type)
device_type(tensor::DLTensor) = device_type(tensor.ctx)
device_type(manager::DLManagedTensor) = device_type(manager.dl_tensor)
device_type(array::DLArray) = device_type(array.manager)

Base.eltype(array::DLArray{T}) where {T} = T

Base.ndims(array::DLArray{T, N}) where {T, N} = N

_size(tensor::DLTensor) = tensor.shape
_size(manager::DLManagedTensor) = _size(manager.dl_tensor)
_size(array::DLArray) = _size(array.manager)

function Base.size(array::DLArray{T, N}) where {T, N}
    ptr = Base.unsafe_convert(Ptr{NTuple{N, Int64}}, _size(array))
    return unsafe_load(ptr)
end
#
function Base.size(array::DLArray{T, N}, d::Integer) where {T, N}
    if 1 ≤ d
        return d ≤ N ? size(array)[d] : Int64(1)
    end
    throw(ArgumentError("Dimension out of range"))
end

_strides(tensor::DLTensor) = tensor.strides
_strides(manager::DLManagedTensor) = _strides(manager.dl_tensor)
_strides(array::DLArray) = _strides(array.manager)

function Base.strides(array::DLArray{T, N}) where {T, N}
    ptr = Base.unsafe_convert(Ptr{NTuple{N, Int64}}, _strides(array))
    return unsafe_load(ptr)
end

byte_offset(tensor::DLTensor) = Int(tensor.byte_offset)
byte_offset(manager::DLManagedTensor) = byte_offset(manager.dl_tensor)
byte_offset(array::DLArray) = byte_offset(array.manager)

Base.pointer(tensor::DLTensor) = tensor.data
Base.pointer(manager::DLManagedTensor) = pointer(manager.dl_tensor)
Base.pointer(array::DLArray) = pointer(array.manager)

##############
#  Wrappers  #
##############

function Base.unsafe_wrap(::Type{Array}, array::DLArray{T}) where {T}
    if device_type(array) == kDLCPU
        addr = Int(pointer(array))
        return GC.@preserve array unsafe_wrap(Array, Ptr{T}(addr), size(array))
    end
    throw(ArgumentError("Only CPU arrays can be wrapped with Array"))
end

function Base.unsafe_wrap(::Type{CuArray}, array::DLArray{T}) where {T}
    if device_type(array) == kDLGPU
        addr = Int(pointer(array))
        return GC.@preserve array unsafe_wrap(CuArray, CuPtr{T}(addr), size(array))
    end
    throw(ArgumentError("Only GPU arrays can be wrapped with CuArray"))
end


end
