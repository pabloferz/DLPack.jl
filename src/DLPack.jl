module DLPack


##  Dependencies  ##

using CUDA
using Requires


##  Exports  ##

export DLArray, DLVector, DLMatrix, RowMajor, ColMajor


##  Aliases and constants  ##

ArrayOrCuArrayT{T, N} = Union{Type{Array{T, N}}, Type{CuArray{T, N}}}

const PYCAPSULE_NAME = Ref(
    (0x64, 0x6c, 0x74, 0x65, 0x6e, 0x73, 0x6f, 0x72, 0x00)
)
const USED_PYCAPSULE_NAME = Ref(
    (0x75, 0x73, 0x65, 0x64, 0x5f, 0x64, 0x6c, 0x74, 0x65, 0x6e, 0x73, 0x6f, 0x72, 0x00)
)


##  Types  ##

@enum DLDeviceType::Cint begin
    kDLCPU = 1
    kDLCUDA = 2
    kDLCUDAHost = 3
    kDLOpenCL = 4
    kDLVulkan = 7
    kDLMetal = 8
    kDLVPI = 9
    kDLROCM = 10
    kDLROCMHost = 11
    kDLExtDev = 12
    kDLCUDAManaged = 13
    kDLOneAPI = 14
    kDLWebGPU = 15
    kDLHexagon = 16
end

struct DLDevice
    device_type::DLDeviceType
    device_id::Cint
end

@enum DLDataTypeCode::Cuint begin
    kDLInt = 0
    kDLUInt = 1
    kDLFloat = 2
    kDLOpaqueHandle = 3
    kDLBfloat = 4
    kDLComplex = 5
end

struct DLDataType
    code::Cuchar
    bits::Cuchar
    lanes::Cushort
end

struct DLTensor
    data::Ptr{Cvoid}
    ctx::DLDevice
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

    function DLManagedTensor(dlptr::Ptr{DLManagedTensor})
        manager = unsafe_load(dlptr)

        if manager.deleter != C_NULL
            delete = manager -> ccall(manager.deleter, Cvoid, (Ptr{Cvoid},), Ref(manager))
            finalizer(delete, manager)
        end

        return manager
    end
end

abstract type MemoryLayout end

struct ColMajor <: MemoryLayout end
struct RowMajor <: MemoryLayout end

struct DLManager{T, N}
    tensor::DLManagedTensor

    function DLManager(tensor::DLManagedTensor)
        T = dtypes_to_jltypes()[tensor.dl_tensor.dtype]
        N = Int(tensor.dl_tensor.ndim)
        return new{T, N}(tensor)
    end

    function DLManager{T, N}(tensor::DLManagedTensor) where {T, N}
        dlt = tensor.dl_tensor
        if N != (n = dlt.ndim)
            throw(ArgumentError("Dimensionality mismatch, object ndims is $n"))
        elseif jltypes_to_dtypes()[T] !== (D = dlt.dtype)
            throw(ArgumentError("Type mismatch, object dtype is $D"))
        end
        return new{T, N}(tensor)
    end
end

struct DLArray{T, N, A <: AbstractArray{T, N}, F}
    manager::DLManager{T, N}
    foreign::F
    data::A
end

function DLArray(tensor::DLManagedTensor, foreign)
    manager = DLManager(tensor)
    dev = device_type(tensor)
    arr = if dev == kDLCPU
        unsafe_wrap(Array, manager)
    elseif dev == kDLCUDA
        unsafe_wrap(CuArray, manager)
    else
        throw(ArgumentError("Unsupported device"))
    end
    data = is_col_major(manager) ? arr : reversedims(arr)
    return DLArray(manager, foreign, data)
end

function DLArray{T, N}(A::TA, ::Type{M}, tensor::DLManagedTensor, foreign) where {
    T, N, TA <: Union{Type{Array}, Type{CuArray}}, M <: MemoryLayout
}
    col_major = is_col_major(tensor, Val(N))
    if (M === ColMajor && !col_major) || (M === RowMajor && col_major)
        throw(ArgumentError("Memory layout mismatch"))
    end
    manager = DLManager{T, N}(tensor)
    data = reversedims_maybe(M, unsafe_wrap(A, manager))
    return DLArray(manager, foreign, data)
end

const DLVector{T} = DLArray{T, 1}
const DLMatrix{T} = DLArray{T, 2}


##  Utils  ##

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
    Float64 => DLDataType(kDLFloat, 64, 1),
    ComplexF32 => DLDataType(kDLComplex, 64, 1),
    ComplexF64 => DLDataType(kDLComplex, 128, 1),
)

dtypes_to_jltypes() = Dict(
    DLDataType(kDLInt, 8, 1) => Int8,
    DLDataType(kDLInt, 16, 1) => Int16,
    DLDataType(kDLInt, 32, 1) => Int32,
    DLDataType(kDLInt, 64, 1) => Int64,
    DLDataType(kDLUInt, 8, 1) => UInt8,
    DLDataType(kDLUInt, 16, 1) => UInt16,
    DLDataType(kDLUInt, 32, 1) => UInt32,
    DLDataType(kDLUInt, 64, 1) => UInt64,
    DLDataType(kDLFloat, 16, 1) => Float16,
    DLDataType(kDLFloat, 32, 1) => Float32,
    DLDataType(kDLFloat, 64, 1) => Float64,
    DLDataType(kDLComplex, 64, 1) => ComplexF32,
    DLDataType(kDLComplex, 128, 1) => ComplexF64,
)

device_type(ctx::DLDevice) = ctx.device_type
device_type(tensor::DLTensor) = device_type(tensor.ctx)
device_type(tensor::DLManagedTensor) = device_type(tensor.dl_tensor)
device_type(manager::DLManager) = device_type(manager.tensor)

Base.eltype(array::DLArray{T}) where {T} = T

Base.ndims(array::DLArray{T, N}) where {T, N} = N

unsafe_size(manager::DLManager{T, N}) where {T, N} = unsafe_size(manager.tensor, Val(N))
#
function unsafe_size(tensor::DLManagedTensor, ::Val{N}) where {N}
    sz = tensor.dl_tensor.shape
    ptr = Base.unsafe_convert(Ptr{NTuple{N, Int64}}, sz)
    return unsafe_load(ptr)
end

Base.size(array::DLArray) = size(array.data)
Base.size(array::DLArray, d::Integer) = size(array.data, d)

function unsafe_strides(tensor::DLManagedTensor, val::Val{N}) where {N}
    st = tensor.dl_tensor.strides
    if st == C_NULL
        trailing_size = Base.rest(unsafe_size(tensor, val), 2)
        tup = ((reverse ∘ cumprod ∘ reverse)(trailing_size)..., 1)
        return NTuple{N, Int64}(tup)
    end
    ptr = Base.unsafe_convert(Ptr{NTuple{N, Int64}}, st)
    return unsafe_load(ptr)
end

Base.strides(a::DLArray) = strides(a.data)

byte_offset(tensor::DLTensor) = Int(tensor.byte_offset)
byte_offset(tensor::DLManagedTensor) = byte_offset(tensor.dl_tensor)
byte_offset(manager::DLManager) = byte_offset(manager.tensor)

Base.pointer(tensor::DLTensor) = tensor.data
Base.pointer(tensor::DLManagedTensor) = pointer(tensor.dl_tensor)
Base.pointer(manager::DLManager) = pointer(manager.tensor)

function Base.unsafe_wrap(::Type{Array}, manager::DLManager{T}) where {T}
    if device_type(manager) == kDLCPU
        addr = Int(pointer(manager))
        sz = unsafe_size(manager)
        return GC.@preserve manager unsafe_wrap(Array, Ptr{T}(addr), sz)
    end
    throw(ArgumentError("Only CPU arrays can be wrapped with Array"))
end
#
function Base.unsafe_wrap(::Type{CuArray}, manager::DLManager{T}) where {T}
    if device_type(manager) == kDLCUDA
        addr = Int(pointer(manager))
        sz = unsafe_size(manager)
        return GC.@preserve manager unsafe_wrap(CuArray, CuPtr{T}(addr), sz)
    end
    throw(ArgumentError("Only CUDA arrays can be wrapped with CuArray"))
end

function is_col_major(manager::DLManager{T, N})::Bool where {T, N}
    return is_col_major(manager.tensor, Val(N))
end
#
function is_col_major(tensor::DLManagedTensor, val::Val{N})::Bool where {N}
    sz = unsafe_size(tensor, val)
    st = unsafe_strides(tensor, val)
    return N == 0 || prod(sz) == 0 || st == Base.size_to_strides(1, sz...)
end

reversedims_maybe(::Type{RowMajor}, array) = reversedims(array)
reversedims_maybe(::Type{ColMajor}, array) = array

function reversedims(a::AbstractArray)
    return revdimstype(a)(reshape(a, (reverse ∘ size)(a)))
end

function revdimstype(a::A) where {T, N, A <: AbstractArray{T, N}}
    P = ntuple(i -> N + 1 - i, Val(N))
    return PermutedDimsArray{T, N, P, P, A}
end


##  Module initialization  ##

function __init__()

    @require PyCall = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0" begin
        include("pycall.jl")
    end

    @require PythonCall = "6099a3de-0909-46bc-b1f4-468b9a2dfc0d" begin
        include("pythoncall.jl")
    end

end


end  # module DLPack
