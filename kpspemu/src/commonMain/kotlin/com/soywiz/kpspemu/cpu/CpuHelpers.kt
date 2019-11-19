package com.soywiz.kpspemu.cpu

import com.soywiz.dynarek2.*
import com.soywiz.kmem.*
import com.soywiz.korma.math.*
import com.soywiz.kpspemu.util.*
import kotlin.math.*
import com.soywiz.korma.math.isAlmostZero as isAlmostZero1

fun D2ContextPspEmu() = D2Context().apply { registerDyna() }

fun dyna_clz(v: Int): Int = BitUtils.clz(v)
fun dyna_clo(v: Int): Int = BitUtils.clo(v)
fun dyna_ext(RS: Int, POS: Int, SIZE_E: Int): Int = RS.extract(POS, SIZE_E)
fun dyna_ins(RT: Int, RS: Int, POS: Int, SIZE_I: Int): Int = RT.insert(RS, POS, SIZE_I)
fun dyna_movz(RT: Int, RD: Int, RS: Int) = if (RT == 0) RS else RD
fun dyna_movn(RT: Int, RD: Int, RS: Int) = if (RT != 0) RS else RD
fun dyna_seb(v: Int) = BitUtils.seb(v)
fun dyna_seh(v: Int) = BitUtils.seh(v)
fun dyna_wsbh(v: Int) = BitUtils.wsbh(v)
fun dyna_wsbw(v: Int) = BitUtils.wsbw(v)
fun dyna_max(a: Int, b: Int) = kotlin.math.max(a, b)
fun dyna_min(a: Int, b: Int) = kotlin.math.min(a, b)
fun dyna_bitrev32(a: Int): Int = BitUtils.bitrev32(a)
fun dyna_rotr(a: Int, b: Int): Int = BitUtils.rotr(a, b)
fun dyna_sll(RT: Int, RS: Int): Int = RT shl (RS and 0b11111)
fun dyna_sra(RT: Int, RS: Int): Int = RT shr (RS and 0b11111)
fun dyna_srl(RT: Int, RS: Int): Int = RT ushr (RS and 0b11111)

fun dyna_divu_LO(RS: Int, RT: Int): Int = if (RT != 0) RS udiv RT else 0
fun dyna_divu_HI(RS: Int, RT: Int): Int = if (RT != 0) RS urem RT else 0

/*
fun dyna_mul_LO(RS: Int, RT: Int): Int = RS * RT
fun dyna_mul_HI(RS: Int, RT: Int): Int = ((RS.toLong() * RT.toLong()) shr 32).toInt()

fun dyna_mulu_LO(RS: Int, RT: Int): Int = RS.toUInt() * RT.toUInt()
fun dyna_mulu_HI(RS: Int, RT: Int): Int = ((RS.toULong() * RT.toUlong()) ushr 32).toInt()
*/

fun dyna_mult(RS: Int, RT: Int): Long = RS.toLong() * RT.toLong()
//fun dyna_multu(RS: Int, RT: Int): Long = (RS.toULong() * RT.toULong()).toLong()

fun dyna_mult_LO(RS: Int, RT: Int): Int = imul32_64_lo(RS, RT)
fun dyna_multu_LO(RS: Int, RT: Int): Int = umul32_64_lo(RS, RT)
fun dyna_mult_HI(RS: Int, RT: Int): Int = imul32_64_hi(RS, RT)
fun dyna_multu_HI(RS: Int, RT: Int): Int = umul32_64_hi(RS, RT)

fun imul32_64(a: Int, b: Int, out: IntArray) {
    out[0] = imul32_64_lo(a, b)
    out[1] = imul32_64_hi(a, b)
}

fun umul32_64(a: Int, b: Int, out: IntArray) {
    out[0] = umul32_64_lo(a, b)
    out[1] = umul32_64_hi(a, b)
}

fun imul32_64_lo(a: Int, b: Int): Int {
    if (a == 0) return 0
    if (b == 0) return 0
    if ((a >= -32768 && a <= 32767) && (b >= -32768 && b <= 32767)) return a * b
    val doNegate = (a < 0) xor (b < 0)
    var result = umul32_64_lo(abs(a), abs(b))
    if (doNegate) result = result.inv() + 1
    return result
}

fun umul32_64_lo(a: Int, b: Int): Int {
    if (a ult 32767 && b ult 65536) return a * b
    val a00 = a and 0xFFFF
    val a16 = a ushr 16
    val b00 = b and 0xFFFF
    val b16 = b ushr 16
    val c00 = a00 * b00
    var c16 = (c00 ushr 16) + (a16 * b00)
    c16 = (c16 and 0xFFFF) + (a00 * b16)
    return ((c16 and 0xFFFF) shl 16) or (c00 and 0xFFFF)
}

fun imul32_64_hi(a: Int, b: Int): Int {
    if (a == 0) return 0
    if (b == 0) return 0
    if ((a >= -32768 && a <= 32767) && (b >= -32768 && b <= 32767)) return if ((a * b) < 0) -1 else 0

    val doNegate = (a < 0) xor (b < 0)

    var result0 = umul32_64_lo(abs(a), abs(b))
    var result1 = umul32_64_hi(abs(a), abs(b))

    if (doNegate) {
        result0 = result0.inv()
        result1 = result1.inv()
        result0 = (result0 + 1) or 0
        if (result0 == 0) result1 = (result1 + 1) or 0
    }

    return result1
}

fun umul32_64_hi(a: Int, b: Int): Int {
    if (a ult 32767 && b ult 65536) return if ((a * b) < 0) -1 else 0
    val a00 = a and 0xFFFF
    val a16 = a ushr 16
    val b00 = b and 0xFFFF
    val b16 = b ushr 16
    val c00 = a00 * b00
    var c16 = (c00 ushr 16) + (a16 * b00)
    var c32 = c16 ushr 16
    c16 = (c16 and 0xFFFF) + (a00 * b16)
    c32 += c16 ushr 16
    var c48 = c32 ushr 16
    c32 = (c32 and 0xFFFF) + (a16 * b16)
    c48 += c32 ushr 16
    return ((c48 and 0xFFFF) shl 16) or (c32 and 0xFFFF)
}


fun dyna_madd(state: CpuState, RS: Int, RT: Int): Int = 0.also { state.HI_LO += RS.toLong() * RT.toLong() }
fun dyna_maddu(state: CpuState, RS: Int, RT: Int): Int = 0.also { state.HI_LO += RS.unsigned * RT.unsigned }
fun dyna_msub(state: CpuState, RS: Int, RT: Int): Int = 0.also { state.HI_LO -= RS.toLong() * RT.toLong() }
fun dyna_msubu(state: CpuState, RS: Int, RT: Int): Int = 0.also { state.HI_LO -= RS.unsigned * RT.unsigned }

fun dyna_fadd(RS: Float, RT: Float): Float = RS pspAdd RT
fun dyna_fsub(RS: Float, RT: Float): Float = RS pspSub RT

fun dyna_fmul(cpu: CpuState, RS: Float, RT: Float): Float {
    val res = RS * RT
    return if (cpu.fcr31_fs && res.isAlmostZero1()) 0f else res
}

fun dyna_fdiv(RS: Float, RT: Float): Float = RS / RT
fun dyna_fneg(v: Float): Float = -v
fun dyna_fabs(v: Float): Float = kotlin.math.abs(v)
fun dyna_fsqrt(v: Float): Float = kotlin.math.sqrt(v)

fun dyna_syscall(state: CpuState, syscall: Int): Int {
    //(state as CpuState).syscall(syscall)
    try {
        state.syscall(syscall)
        //syscalls.syscall(this, syscall)
        return 0
    } catch (e: CpuBreakException) {
        return e.id
    }
}

fun dyna_break(syscall: Int): Int = throw CpuBreakExceptionCached(syscall)

fun dyna_slt(RS: Int, RT: Int): Int = (RS < RT).toInt()
fun dyna_sltu(RS: Int, RT: Int): Int = (RS ult RT).toInt()

fun dyna_trunc_w_s(v: Float): Int = Math.trunc(v)
fun dyna_round_w_s(v: Float): Int = Math.round(v)
fun dyna_ceil_w_s(v: Float): Int = Math.ceil(v)
fun dyna_floor_w_s(v: Float): Int = Math.floor(v)

fun dyna_checkFNan(cpu: CpuState, FD: Float): Int {
    if (FD.isNaN()) cpu.fcr31 = cpu.fcr31 or 0x00010040
    if (FD.isInfinite()) cpu.fcr31 = cpu.fcr31 or 0x00005014
    return 0
}

fun dyna_cvt_s_w(v: Int): Float = v.toFloat()
fun dyna_cvt_w_s(cpu: CpuState, FS: Float): Int {
    return when (cpu.fcr31_rm) {
        0 -> Math.rint(FS) // rint: round nearest
        1 -> Math.cast(FS) // round to zero
        2 -> Math.ceil(FS) // round up (ceil)
        3 -> Math.floor(FS) // round down (floor)
        else -> FS.toInt()
    }
}

private inline fun _cu(FS: Float, FT: Float, callback: () -> Boolean): Boolean =
    if (FS.isNaN() || FT.isNaN()) true else callback()

private inline fun _co(FS: Float, FT: Float, callback: () -> Boolean): Boolean =
    if (FS.isNaN() || FT.isNaN()) false else callback()

fun dyna_c_f_s(FS: Float, FT: Float) = _co(FS, FT) { false }
fun dyna_c_un_s(FS: Float, FT: Float) = _cu(FS, FT) { false }
fun dyna_c_eq_s(FS: Float, FT: Float) = _co(FS, FT) { FS == FT }
fun dyna_c_ueq_s(FS: Float, FT: Float) = _cu(FS, FT) { FS == FT }
fun dyna_c_olt_s(FS: Float, FT: Float) = _co(FS, FT) { FS < FT }
fun dyna_c_ult_s(FS: Float, FT: Float) = _cu(FS, FT) { FS < FT }
fun dyna_c_ole_s(FS: Float, FT: Float) = _co(FS, FT) { FS <= FT }
fun dyna_c_ule_s(FS: Float, FT: Float) = _cu(FS, FT) { FS <= FT }
