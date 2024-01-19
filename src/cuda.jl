# SPDX-License-Identifier: MIT
# See LICENSE.md at https://github.com/pabloferz/DLPack.jl

buftype(x::CUDA.CuArray) = buftype(typeof(x))
buftype(::Type{<:CUDA.CuArray{<:Any, <:Any, B}}) where {B} = @isdefined(B) ? B : Any

share(A::CUDA.StridedCuArray) = unsafe_share(parent(A))

jlarray_type(::Val{kDLCUDA}) = CUDA.CuArray
jlarray_type(::Val{kDLCUDAHost}) = CUDA.CuArray
jlarray_type(::Val{kDLCUDAManaged}) = CUDA.CuArray

function dldevice(x::CUDA.StridedCuArray)
    y = parent(x)
    B = buftype(y)

    dldt = if B === CUDA.Mem.DeviceBuffer
        kDLCUDA
    elseif B === CUDA.Mem.HostBuffer
        kDLCUDAHost
    elseif B === CUDA.Mem.UnifiedBuffer
        kDLCUDAManaged
    end

    return DLDevice(dldt, CUDA.device(y))
end

function Base.unsafe_wrap(::Type{<: CUDA.CuArray}, manager::DLManager{T}) where {T}
    if device_type(manager) == kDLCUDA
        addr = Int(pointer(manager))
        sz = unsafe_size(manager)
        return unsafe_wrap(CUDA.CuArray, CUDA.CuPtr{T}(addr), sz)
    end
    throw(ArgumentError("Only CUDA arrays can be wrapped with CuArray"))
end
