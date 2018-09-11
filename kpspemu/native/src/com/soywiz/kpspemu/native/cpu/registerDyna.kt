package com.soywiz.kpspemu.cpu

import kotlinx.cinterop.*
import platform.posix.*
import kotlin.reflect.*
import com.soywiz.dynarek2.*
import com.soywiz.kpspemu.*
import com.soywiz.kpspemu.cpu.*

val COpaquePointer.asCpuState get() = this.asStableRef<CpuState>().get()

fun _dyna_syscall(ptr: COpaquePointer, syscall: Int): Int = dyna_syscall(ptr.asCpuState, syscall)
fun _dyna_checkFNan(ptr: COpaquePointer, FD: Float): Int = dyna_checkFNan(ptr.asCpuState, FD)
fun _dyna_fmul(ptr: COpaquePointer, RS: Float, RT: Float): Float = dyna_fmul(ptr.asCpuState, RS, RT)
fun _dyna_cvt_w_s(ptr: COpaquePointer, FS: Float): Int = dyna_cvt_w_s(ptr.asCpuState, FS)

actual fun D2Context.registerDyna(): Unit {
    registerFunc(::iprint, staticCFunction(::iprint).asLong)

    // Plain
    registerFunc(::dyna_clz, staticCFunction(::dyna_clz).asLong)
    registerFunc(::dyna_clo, staticCFunction(::dyna_clo).asLong)
    registerFunc(::dyna_ext, staticCFunction(::dyna_ext).asLong)
    registerFunc(::dyna_ins, staticCFunction(::dyna_ins).asLong)
    registerFunc(::dyna_movz, staticCFunction(::dyna_movz).asLong)
    registerFunc(::dyna_movn, staticCFunction(::dyna_movn).asLong)
    registerFunc(::dyna_seb, staticCFunction(::dyna_seb).asLong)
    registerFunc(::dyna_seh, staticCFunction(::dyna_seh).asLong)
    registerFunc(::dyna_wsbh, staticCFunction(::dyna_wsbh).asLong)
    registerFunc(::dyna_wsbw, staticCFunction(::dyna_wsbw).asLong)
    registerFunc(::dyna_max, staticCFunction(::dyna_max).asLong)
    registerFunc(::dyna_min, staticCFunction(::dyna_min).asLong)
    registerFunc(::dyna_bitrev32, staticCFunction(::dyna_bitrev32).asLong)
    registerFunc(::dyna_rotr, staticCFunction(::dyna_rotr).asLong)
    registerFunc(::dyna_sll, staticCFunction(::dyna_sll).asLong)
    registerFunc(::dyna_sra, staticCFunction(::dyna_sra).asLong)
    registerFunc(::dyna_srl, staticCFunction(::dyna_srl).asLong)
    registerFunc(::dyna_divu_LO, staticCFunction(::dyna_divu_LO).asLong)
    registerFunc(::dyna_divu_HI, staticCFunction(::dyna_divu_HI).asLong)
    registerFunc(::dyna_mult, staticCFunction(::dyna_mult).asLong)
    registerFunc(::dyna_mult_LO, staticCFunction(::dyna_mult_LO).asLong)
    registerFunc(::dyna_multu_LO, staticCFunction(::dyna_multu_LO).asLong)
    registerFunc(::dyna_mult_HI, staticCFunction(::dyna_mult_HI).asLong)
    registerFunc(::dyna_multu_HI, staticCFunction(::dyna_multu_HI).asLong)
    registerFunc(::dyna_madd, staticCFunction(::dyna_madd).asLong)
    registerFunc(::dyna_maddu, staticCFunction(::dyna_maddu).asLong)
    registerFunc(::dyna_msub, staticCFunction(::dyna_msub).asLong)
    registerFunc(::dyna_msubu, staticCFunction(::dyna_msubu).asLong)
    registerFunc(::dyna_fadd, staticCFunction(::dyna_fadd).asLong)
    registerFunc(::dyna_fsub, staticCFunction(::dyna_fsub).asLong)
    registerFunc(::dyna_fdiv, staticCFunction(::dyna_fdiv).asLong)
    registerFunc(::dyna_fneg, staticCFunction(::dyna_fneg).asLong)
    registerFunc(::dyna_fabs, staticCFunction(::dyna_fabs).asLong)
    registerFunc(::dyna_fsqrt, staticCFunction(::dyna_fsqrt).asLong)
    registerFunc(::dyna_break, staticCFunction(::dyna_break).asLong)
    registerFunc(::dyna_slt, staticCFunction(::dyna_slt).asLong)
    registerFunc(::dyna_sltu, staticCFunction(::dyna_sltu).asLong)
    registerFunc(::dyna_trunc_w_s, staticCFunction(::dyna_trunc_w_s).asLong)
    registerFunc(::dyna_round_w_s, staticCFunction(::dyna_round_w_s).asLong)
    registerFunc(::dyna_ceil_w_s, staticCFunction(::dyna_ceil_w_s).asLong)
    registerFunc(::dyna_floor_w_s, staticCFunction(::dyna_floor_w_s).asLong)
    registerFunc(::dyna_cvt_s_w, staticCFunction(::dyna_cvt_s_w).asLong)
    registerFunc(::dyna_c_f_s, staticCFunction(::dyna_c_f_s).asLong)
    registerFunc(::dyna_c_un_s, staticCFunction(::dyna_c_un_s).asLong)
    registerFunc(::dyna_c_eq_s, staticCFunction(::dyna_c_eq_s).asLong)
    registerFunc(::dyna_c_ueq_s, staticCFunction(::dyna_c_ueq_s).asLong)
    registerFunc(::dyna_c_olt_s, staticCFunction(::dyna_c_olt_s).asLong)
    registerFunc(::dyna_c_ult_s, staticCFunction(::dyna_c_ult_s).asLong)
    registerFunc(::dyna_c_ole_s, staticCFunction(::dyna_c_ole_s).asLong)
    registerFunc(::dyna_c_ule_s, staticCFunction(::dyna_c_ule_s).asLong)

    // With EXTERNAL CpuState
    registerFunc(::dyna_fmul, staticCFunction(::_dyna_fmul).asLong)
    registerFunc(::dyna_syscall, staticCFunction(::_dyna_syscall).asLong)
    registerFunc(::dyna_checkFNan, staticCFunction(::_dyna_checkFNan).asLong)
    registerFunc(::dyna_cvt_w_s, staticCFunction(::_dyna_cvt_w_s).asLong)
}
