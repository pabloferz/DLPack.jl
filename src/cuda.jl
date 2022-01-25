function Base.unsafe_wrap(::Type{CUDA.CuArray}, array::DLArray{T}) where {T}
    if device_type(array) == kDLCUDA
        addr = Int(pointer(array))
        return GC.@preserve array unsafe_wrap(
            CUDA.CuArray, CUDA.CuPtr{T}(addr), size(array)
        )
    end
    throw(ArgumentError("Only CUDA arrays can be wrapped with CuArray"))
end
