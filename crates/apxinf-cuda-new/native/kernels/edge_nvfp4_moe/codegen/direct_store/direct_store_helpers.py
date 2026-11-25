"""Private expanded-route TopK1 BF16 output only. Not generic token-level MoE reduction."""
import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

@dsl_user_op
def normalize_bf16_add_zero_inplace(r_c, *, loc=None, ip=None):
    # Reinterpret bits of the already rounded BF16 registers; never numeric-cast
    # to integer/FP32, which could change NaN payloads or subnormal behavior.
    words = cute.recast_tensor(r_c, cutlass.Uint32, loc=loc, ip=ip)
    count = cute.size(words.shape)
    for index in range(count):
        value = words[index]
        result = llvm.inline_asm(
            T.i32(), [value.ir_value(), cutlass.Uint32(0).ir_value()],
            "add.rn.bf16x2 $0, $1, $2;", "=r,r,r",
            has_side_effects=True, loc=loc, ip=ip,
        )
        words[index] = cutlass.Uint32(result)

@dsl_user_op
def blk_store_bf16(dst_gemm, src_smem, size, loc=None, ip=None):
    # Caller retains original fence/proxy/commit/wait/barrier protocol and row
    # alignment/length. Requires exclusive expanded-route output ownership.
    llvm.inline_asm(
        None, [dst_gemm.iterator.llvm_ptr, src_smem.iterator.llvm_ptr, size.ir_value()],
        "cp.async.bulk.global.shared::cta.bulk_group [$0], [$1], $2;",
        "l,l,r", has_side_effects=True, loc=loc, ip=ip,
    )
