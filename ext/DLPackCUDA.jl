# SPDX-License-Identifier: MIT
# See LICENSE.md at https://github.com/pabloferz/DLPack.jl

module DLPackCUDA


##  Dependencies  ##

import DLPack
@static if isdefined(Base, :get_extension)
    import CUDA
else
    import ..CUDA
end


##  Extensions  ##

buftype(x::CUDA.CuArray) = buftype(typeof(x))
buftype(::Type{<:CUDA.CuArray{<:Any, <:Any, B}}) where {B} = @isdefined(B) ? B : Any

DLPack.share(A::CUDA.StridedCuArray) = DLPack.unsafe_share(parent(A))

DLPack.jlarray_type(::Val{DLPack.kDLCUDA}) = CUDA.CuArray
DLPack.jlarray_type(::Val{DLPack.kDLCUDAHost}) = CUDA.CuArray
DLPack.jlarray_type(::Val{DLPack.kDLCUDAManaged}) = CUDA.CuArray

function DLPack.dldevice(x::CUDA.StridedCuArray)
    y = parent(x)
    B = buftype(y)

    dldt = if B === CUDA.Mem.DeviceBuffer
        DLPack.kDLCUDA
    elseif B === CUDA.Mem.HostBuffer
        DLPack.kDLCUDAHost
    elseif B === CUDA.Mem.UnifiedBuffer
        DLPack.kDLCUDAManaged
    end

    return DLPack.DLDevice(dldt, CUDA.device(y))
end

function Base.unsafe_wrap(::Type{<: CUDA.CuArray}, manager::DLPack.DLManager{T}) where {T}
    if DLPack.device_type(manager) == DLPack.kDLCUDA
        addr = Int(pointer(manager))
        sz = DLPack.unsafe_size(manager)
        return DLPack.unsafe_wrap(CUDA.CuArray, CUDA.CuPtr{T}(addr), sz)
    end
    throw(ArgumentError("Only CUDA arrays can be wrapped with CuArray"))
end


end  # module DLPackCUDA
