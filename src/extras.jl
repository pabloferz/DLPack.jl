# SPDX-License-Identifier: MIT
# See LICENSE.md at https://github.com/pabloferz/DLPack.jl

let
    global ColMajor
    global RowMajor

    layout_docstring = t -> (
        "Used as tag to indicate that the memory layout of an imported tensor is $t-major"
    )

    Docs.getdoc(::Type{ColMajor}) = layout_docstring("col")
    Docs.getdoc(::Type{RowMajor}) = layout_docstring("row")
end
