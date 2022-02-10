module DLPack


##  Dependencies  ##

using Requires


##  Exports  ##

export DLArray, DLVector, DLMatrix, RowMajor, ColMajor


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

    function DLManagedTensor(
        dl_tensor::DLTensor, manager_ctx::Ptr{Cvoid}, deleter::Ptr{Cvoid}
    )
        return new(dl_tensor, manager_ctx, deleter)
    end

    function DLManagedTensor(dlptr::Ptr{DLManagedTensor})
        manager = unsafe_load(dlptr)

        if manager.deleter != C_NULL
            delete = manager -> ccall(manager.deleter, Cvoid, (Ptr{Cvoid},), Ref(manager))
            finalizer(delete, manager)
        end

        return manager
    end
end

struct Capsule
    tensor::DLManagedTensor
    shape::Vector{Clonglong}
    strides::Vector{Clonglong}
end

share(A::StridedArray) = unsafe_share(A)

function unsafe_share(A::AbstractArray{T, N}) where {T, N}
    sh = Clonglong[(reverse ∘ size)(A)...]
    st = Clonglong[(reverse ∘ strides)(A)...]

    # This should work for Ptr and CuPtr
    data = Ptr{Cvoid}(UInt(pointer(A)))
    ctx = dldevice(A)
    ndim = Cint(N)
    dtype = jltypes_to_dtypes()[T]
    sh_ptr = pointer(sh)
    st_ptr = pointer(st)
    dl_tensor = DLTensor(data, ctx, ndim, dtype, sh_ptr, st_ptr, Culonglong(0))
    tensor = DLManagedTensor(dl_tensor, C_NULL, C_NULL)

    return Capsule(tensor, sh, st)
end

abstract type MemoryLayout end

struct ColMajor <: MemoryLayout end
struct RowMajor <: MemoryLayout end

struct DLManager{T, N}
    manager::DLManagedTensor

    function DLManager(manager::DLManagedTensor)
        T = dtypes_to_jltypes()[manager.dl_tensor.dtype]
        N = Int(manager.dl_tensor.ndim)
        return new{T, N}(manager)
    end

    function DLManager{T, N}(manager::DLManagedTensor) where {T, N}
        dlt = manager.dl_tensor
        if N != (n = dlt.ndim)
            throw(ArgumentError("Dimensionality mismatch, object ndims is $n"))
        elseif jltypes_to_dtypes()[T] !== (D = dlt.dtype)
            throw(ArgumentError("Type mismatch, object dtype is $D"))
        end
        return new{T, N}(manager)
    end
end

struct DLArray{T, N, A <: AbstractArray{T, N}, F} <: AbstractArray{T, N}
    foreign::F
    data::A
end

function DLArray(manager::DLManagedTensor, foreign)
    GC.@preserve manager begin
        typed_manager = DLManager(manager)
        A = jlarray_type(Val(device_type(manager)))
        arr = unsafe_wrap(A, typed_manager)
        data = is_col_major(typed_manager) ? arr : reversedims(arr)
    end
    return DLArray(foreign, data)
end

function DLArray{T, N}(::Type{A}, ::Type{M}, manager::DLManagedTensor, foreign) where {
    T, N, A, M <: MemoryLayout
}
    col_major = is_col_major(manager, Val(N))
    if (M === ColMajor && !col_major) || (M === RowMajor && col_major)
        throw(ArgumentError("Memory layout mismatch"))
    end
    typed_manager = DLManager{T, N}(manager)
    data = reversedims_maybe(M, unsafe_wrap(A, typed_manager))
    return DLArray(foreign, data)
end


##  Aliases and constants  ##

const DLVector{T} = DLArray{T, 1}
const DLMatrix{T} = DLArray{T, 2}

const PYCAPSULE_NAME = Ref(
    (0x64, 0x6c, 0x74, 0x65, 0x6e, 0x73, 0x6f, 0x72, 0x00)
)

const USED_PYCAPSULE_NAME = Ref(
    (0x75, 0x73, 0x65, 0x64, 0x5f, 0x64, 0x6c, 0x74, 0x65, 0x6e, 0x73, 0x6f, 0x72, 0x00)
)

const DLPACK_POOL = Dict{Ptr{Cvoid}, Tuple{Capsule, Any}}()


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

jlarray_type(::Val{kDLCPU}) = Array
#
function jlarray_type(::Val{D}) where {D}
    if D in (kDLCUDA, kDLCUDAHost, kDLCUDAManaged)
        throw("CUDA package is not loaded")
    else
        throw("Unsupported device")
    end
end

dldevice(::StridedArray) = DLDevice(kDLCPU, Cint(0))

device_type(ctx::DLDevice) = ctx.device_type
device_type(tensor::DLTensor) = device_type(tensor.ctx)
device_type(manager::DLManagedTensor) = device_type(manager.dl_tensor)
device_type(manager::DLManager) = device_type(manager.manager)

unsafe_size(manager::DLManager{T, N}) where {T, N} = unsafe_size(manager.manager, Val(N))
#
function unsafe_size(manager::DLManagedTensor, ::Val{N}) where {N}
    sz = manager.dl_tensor.shape
    ptr = Base.unsafe_convert(Ptr{NTuple{N, Int64}}, sz)
    return unsafe_load(ptr)
end

function unsafe_strides(manager::DLManagedTensor, val::Val{N}) where {N}
    st = manager.dl_tensor.strides
    if st == C_NULL
        sz = unsafe_size(manager, val)
        tup = ((reverse ∘ cumprod ∘ reverse ∘ Base.tail)(sz)..., 1)
        return NTuple{N, Int64}(tup)
    end
    ptr = Base.unsafe_convert(Ptr{NTuple{N, Int64}}, st)
    return unsafe_load(ptr)
end

byte_offset(tensor::DLTensor) = Int(tensor.byte_offset)
byte_offset(manager::DLManagedTensor) = byte_offset(manager.dl_tensor)
byte_offset(manager::DLManager) = byte_offset(manager.manager)

Base.pointer(tensor::DLTensor) = tensor.data
Base.pointer(manager::DLManagedTensor) = pointer(manager.dl_tensor)
Base.pointer(manager::DLManager) = pointer(manager.manager)

function Base.unsafe_wrap(::Type{Array}, manager::DLManager{T}) where {T}
    if device_type(manager) == kDLCPU
        addr = Int(pointer(manager))
        sz = unsafe_size(manager)
        return unsafe_wrap(Array, Ptr{T}(addr), sz)
    end
    throw(ArgumentError("Only CPU arrays can be wrapped with Array"))
end

function is_col_major(manager::DLManager{T, N})::Bool where {T, N}
    return is_col_major(manager.manager, Val(N))
end
#
function is_col_major(manager::DLManagedTensor, val::Val{N})::Bool where {N}
    sz = unsafe_size(manager, val)
    st = unsafe_strides(manager, val)
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

function release(ptr)
    delete!(DLPACK_POOL, ptr)
    return nothing
end


##  Array Interface  ##

Base.size(A::DLArray) = size(A.data)
Base.size(A::DLArray, d::Integer) = size(A.data, d)

Base.@propagate_inbounds Base.getindex(A::DLArray, I...) = getindex(A.data, I...)

Base.@propagate_inbounds Base.setindex!(A::DLArray, v, I...) = setindex!(A.data, v, I...)

Base.strides(a::DLArray) = strides(a.data)

function Base.unsafe_convert(::Type{Ptr{T}}, A::DLArray) where {T}
    return Base.unsafe_convert(Ptr{T}, A.data)
end

Base.elsize(::Type{D}) where {T, N, A, D <: DLArray{T, N, A}} = Base.elsize(A)

function Base.Broadcast.BroadcastStyle(::Type{D}) where {T, N, A, D <: DLArray{T, N, A}}
    return Base.Broadcast.BroadcastStyle(A)
end


##  Module initialization  ##

function __init__()

    @require CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba" begin
        include("cuda.jl")
    end

    @require PyCall = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0" begin
        include("pycall.jl")
    end

    @require PythonCall = "6099a3de-0909-46bc-b1f4-468b9a2dfc0d" begin
        include("pythoncall.jl")
    end

end


end  # module DLPack
