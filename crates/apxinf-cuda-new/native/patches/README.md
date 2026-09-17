# Vendor patches

Vendor source under `native/kernels/fa2/flash_attn` and
`native/kernels/cutlass/fmha` is kept byte-identical to its documented upstream
snapshot whenever possible. Any unavoidable source-level change must be stored
as a reviewable patch in this directory rather than silently edited in the
vendor tree.

Each patch must document its upstream revision, affected files, reason, and
validation. Compatibility shims belong in an ApxInf-owned sibling directory,
and architecture dispatch belongs in an operator wrapper. The guarded FA2
direct-E4M3 conversion is the only current vendor-header exception; its scale
is passed through the upstream parameter ABI by the operator translation unit.
