share(A::CUDA.StridedCuArray) = unsafe_share(parent(A))

jlarray_type(::Val{kDLCUDA}) = CUDA.CuArray
jlarray_type(::Val{kDLCUDAHost}) = CUDA.CuArray
jlarray_type(::Val{kDLCUDAManaged}) = CUDA.CuArray

function dldevice(B::CUDA.StridedCuArray)
    A = parent(B)
    buf = A.storage.buffer

    dldt = if buf isa CUDA.Mem.DeviceBuffer
        kDLCUDA
    elseif buf isa CUDA.Mem.HostBuffer
        kDLCUDAHost
    elseif buf isa CUDA.Mem.UnifiedBuffer
        kDLCUDAManaged
    end

    return DLDevice(dldt, CUDA.device(A))
end

function Base.unsafe_wrap(::Type{CUDA.CuArray}, manager::DLManager{T}) where {T}
    if device_type(manager) == kDLCUDA
        addr = Int(pointer(manager))
        sz = unsafe_size(manager)
        return unsafe_wrap(CUDA.CuArray, CUDA.CuPtr{T}(addr), sz)
    end
    throw(ArgumentError("Only CUDA arrays can be wrapped with CuArray"))
end
