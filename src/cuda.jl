jlarray_type(::Val{kDLCUDA}) = CUDA.CuArray
jlarray_type(::Val{kDLCUDAHost}) = CUDA.CuArray
jlarray_type(::Val{kDLCUDAManaged}) = CUDA.CuArray

function Base.unsafe_wrap(::Type{CUDA.CuArray}, manager::DLManager{T}) where {T}
    if device_type(manager) == kDLCUDA
        addr = Int(pointer(manager))
        sz = unsafe_size(manager)
        return GC.@preserve manager unsafe_wrap(CUDA.CuArray, CUDA.CuPtr{T}(addr), sz)
    end
    throw(ArgumentError("Only CUDA arrays can be wrapped with CuArray"))
end
