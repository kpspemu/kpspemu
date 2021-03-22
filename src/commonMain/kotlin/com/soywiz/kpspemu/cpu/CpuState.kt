package com.soywiz.kpspemu.cpu

import com.soywiz.kds.*
import com.soywiz.kmem.*
import com.soywiz.korge.util.NativeThreadLocal
import com.soywiz.korio.lang.*
import com.soywiz.korio.util.*
import com.soywiz.korma.math.*
import com.soywiz.kpspemu.mem.*
import com.soywiz.kpspemu.util.*
import com.soywiz.krypto.encoding.*
import com.soywiz.korio.error.invalidOp as invalidOp1
import com.soywiz.korma.math.isAlmostZero as isAlmostZero1

abstract class EmulatorControlFlowException : Exception()

data class BreakpointException(val cpu: CpuState, val pc: Int) : EmulatorControlFlowException()

data class CpuBreakException(val id: Int) : EmulatorControlFlowException() {
    companion object {
        val THREAD_WAIT = 10001
        val THREAD_EXIT_KILL = 10002
        val INTERRUPT_RETURN = 10003

        val THREAD_WAIT_RA = MemoryInfo.MAIN_OFFSET + 0
        val THREAD_EXIT_KIL_RA = MemoryInfo.MAIN_OFFSET + 4
        val INTERRUPT_RETURN_RA = MemoryInfo.MAIN_OFFSET + 8

        fun initialize(mem: Memory) {
            mem.sw(THREAD_WAIT_RA, 0b000000_00000000000000000000_001101 or (CpuBreakException.THREAD_WAIT shl 6))
            mem.sw(
                THREAD_EXIT_KIL_RA,
                0b000000_00000000000000000000_001101 or (CpuBreakException.THREAD_EXIT_KILL shl 6)
            )
            mem.sw(
                INTERRUPT_RETURN_RA,
                0b000000_00000000000000000000_001101 or (CpuBreakException.INTERRUPT_RETURN shl 6)
            )
        }
    }
}

//private val CpuBreakExceptionWait = CpuBreakException(CpuBreakException.THREAD_WAIT)
//private val CpuBreakExceptionThreadExitKill = CpuBreakException(CpuBreakException.THREAD_EXIT_KILL)
//private val CpuBreakExceptionInterruptReturn = CpuBreakException(CpuBreakException.INTERRUPT_RETURN)

private val CpuBreakExceptionWait get() = CpuBreakException(CpuBreakException.THREAD_WAIT)
private val CpuBreakExceptionThreadExitKill get() = CpuBreakException(CpuBreakException.THREAD_EXIT_KILL)
private val CpuBreakExceptionInterruptReturn get() = CpuBreakException(CpuBreakException.INTERRUPT_RETURN)

fun CpuBreakExceptionCached(id: Int): CpuBreakException {
    // JS Can't throw cached exceptions
    if (OS.isJs) return CpuBreakException(id)

    return when (id) {
        CpuBreakException.THREAD_WAIT -> CpuBreakExceptionWait
        CpuBreakException.THREAD_EXIT_KILL -> CpuBreakExceptionThreadExitKill
        CpuBreakException.INTERRUPT_RETURN -> CpuBreakExceptionInterruptReturn
        else -> CpuBreakException(id)
    }
}

// http://www.cs.uwm.edu/classes/cs315/Bacon/Lecture/HTML/ch05s03.html
var CpuState.K0: Int; set(value) = run { r26 = value }; get() = r26
var CpuState.K1: Int; set(value) = run { r27 = value }; get() = r27
var CpuState.GP: Int; set(value) = run { r28 = value }; get() = r28
var CpuState.SP: Int; set(value) = run { r29 = value }; get() = r29
var CpuState.FP: Int; set(value) = run { r30 = value }; get() = r30
var CpuState.RA: Int; set(value) = run { r31 = value }; get() = r31

var CpuState.V0: Int; set(value) = run { r2 = value }; get() = r2
var CpuState.V1: Int; set(value) = run { r3 = value }; get() = r3
var CpuState.A0: Int; set(value) = run { r4 = value }; get() = r4
var CpuState.A1: Int; set(value) = run { r5 = value }; get() = r5
var CpuState.A2: Int; set(value) = run { r6 = value }; get() = r6
var CpuState.A3: Int; set(value) = run { r7 = value }; get() = r7

data class RegInfo(val index: Int, val name: String, val mnemonic: String, val desc: String)

class CpuState(
    val name: String,
    val globalCpuState: GlobalCpuState,
    val syscalls: Syscalls = TraceSyscallHandler()
) : Extra by Extra.Mixin() {
    val mem: Memory = globalCpuState.mem

    @NativeThreadLocal
    companion object {
        val gprInfos = listOf(
            RegInfo(0, "r0", "zero", "Permanently 0"),
            RegInfo(1, "r1", "at", "Assembler Temporaty"),
            RegInfo(2, "r2", "v0", "Value returned by a subroutine"),
            RegInfo(3, "r3", "v1", "Value returned by a subroutine"),
            RegInfo(4, "r4", "a0", "Subroutine Arguments"),
            RegInfo(5, "r5", "a1", "Subroutine Arguments"),
            RegInfo(6, "r6", "a2", "Subroutine Arguments"),
            RegInfo(7, "r7", "a3", "Subroutine Arguments"),
            RegInfo(8, "r8", "t0", "Temporary"),
            RegInfo(9, "r9", "t1", "Temporary"),
            RegInfo(10, "r10", "t2", "Temporary"),
            RegInfo(11, "r11", "t3", "Temporary"),
            RegInfo(12, "r12", "t4", "Temporary"),
            RegInfo(13, "r13", "t5", "Temporary"),
            RegInfo(14, "r14", "t6", "Temporary"),
            RegInfo(15, "r15", "t7", "Temporary"),
            RegInfo(16, "r16", "s0", "Saved registers"),
            RegInfo(17, "r17", "s1", "Saved registers"),
            RegInfo(18, "r18", "s2", "Saved registers"),
            RegInfo(19, "r19", "s3", "Saved registers"),
            RegInfo(20, "r20", "s4", "Saved registers"),
            RegInfo(21, "r21", "s5", "Saved registers"),
            RegInfo(22, "r22", "s6", "Saved registers"),
            RegInfo(23, "r23", "s7", "Saved registers"),
            RegInfo(24, "r24", "t8", "Temporary"),
            RegInfo(25, "r25", "t9", "Temporary"),
            RegInfo(26, "r26", "k0", "Kernel"),
            RegInfo(27, "r27", "k1", "Kernel"),
            RegInfo(28, "r28", "gp", "Global Pointer"),
            RegInfo(29, "r29", "sp", "Stack Pointer"),
            RegInfo(30, "r30", "fp", "Frame Pointer"),
            RegInfo(31, "r31", "fp", "Return Address")
        )

        val gprInfosByMnemonic = (gprInfos.map { it.mnemonic to it } + gprInfos.map { it.name to it }).toMap()

        val dummy = CpuState("dummy", GlobalCpuState.dummy)

        var lastId = 0

        fun getGprProp(index: Int) = when (index) {
            0 -> CpuState::r0;1 -> CpuState::r1;2 -> CpuState::r2;3 -> CpuState::r3;4 -> CpuState::r4;5 -> CpuState::r5;6 -> CpuState::r6;7 -> CpuState::r7;
            8 -> CpuState::r8;9 -> CpuState::r9;10 -> CpuState::r10;11 -> CpuState::r11;12 -> CpuState::r12;13 -> CpuState::r13;14 -> CpuState::r14;15 -> CpuState::r15;
            16 -> CpuState::r16;17 -> CpuState::r17;18 -> CpuState::r18;19 -> CpuState::r19;20 -> CpuState::r20;21 -> CpuState::r21;22 -> CpuState::r22;23 -> CpuState::r23;
            24 -> CpuState::r24;25 -> CpuState::r25;26 -> CpuState::r26;27 -> CpuState::r27;28 -> CpuState::r28;29 -> CpuState::r29;30 -> CpuState::r30;31 -> CpuState::r31;
            else -> invalidArg("Invalid register $index")
        }
    }

    fun getGprProp(index: Int) = when (index) {
        0 -> ::r0;1 -> ::r1;2 -> ::r2;3 -> ::r3;4 -> ::r4;5 -> ::r5;6 -> ::r6;7 -> ::r7;
        8 -> ::r8;9 -> ::r9;10 -> ::r10;11 -> ::r11;12 -> ::r12;13 -> ::r13;14 -> ::r14;15 -> ::r15;
        16 -> ::r16;17 -> ::r17;18 -> ::r18;19 -> ::r19;20 -> ::r20;21 -> ::r21;22 -> ::r22;23 -> ::r23;
        24 -> ::r24;25 -> ::r25;26 -> ::r26;27 -> ::r27;28 -> ::r28;29 -> ::r29;30 -> ::r30;31 -> ::r31;
        else -> invalidArg("Invalid register $index")
    }

    val id = lastId++
    var totalExecuted: Long = 0L

    val _FMem = MemBufferAlloc(32 * 4)
    var _F = _FMem.asFloat32Buffer()
    var _FI = _FMem.asInt32Buffer()

    val _VFPRMem = MemBufferAlloc(128 * 4)
    val _VFPR = _VFPRMem.asFloat32Buffer().apply {
        for (n in 0 until 128) this[n] = Float.NaN
    }
    val _VFPR_I = _VFPRMem.asInt32Buffer()
    val VFPRC = intArrayOf(
        0,
        0,
        0,
        0xFF,
        0,
        0,
        0,
        0,
        0x3F800000,
        0x3F800000,
        0x3F800000,
        0x3F800000,
        0x3F800000,
        0x3F800000,
        0x3F800000,
        0x3F800000
    )
    var VFPR_CC: Int get() = VFPRC[CpuState.VFPU_CTRL.CC]; set(value) = run { VFPRC[CpuState.VFPU_CTRL.CC] = value }
    fun VFPR_CC(index: Int) = VFPR_CC.extract(index)

    var fcr0: Int = 0x00003351
    var fcr25: Int = 0
    var fcr26: Int = 0
    var fcr27: Int = 0
    var fcr28: Int = 0
    var fcr31: Int = 0x00000e00

    var vpfxs = VfpuSourceTargetPrefix(VFPRC, CpuState.VFPU_CTRL.SPREFIX)
    var vpfxt = VfpuSourceTargetPrefix(VFPRC, CpuState.VFPU_CTRL.TPREFIX)
    var vpfxd = VfpuDestinationPrefix(VFPRC, CpuState.VFPU_CTRL.DPREFIX)

    fun updateFCR31(value: Int) {
        fcr31 = value and 0x0183FFFF
    }

    var fcr31_rm: Int set(value) = run { fcr31 = fcr31.insert(value, 0, 2) }; get() = fcr31.extract(0, 2)
    var fcr31_2_21: Int set(value) = run { fcr31 = fcr31.insert(value, 2, 21) }; get() = fcr31.extract(2, 21)
    var fcr31_cc: Boolean set(value) = run { fcr31 = fcr31.insert(value, 23) }; get() = fcr31.extract(23)
    var fcr31_fs: Boolean set(value) = run { fcr31 = fcr31.insert(value, 24) }; get() = fcr31.extract(24)
    var fcr31_25_7: Int set(value) = run { fcr31 = fcr31.insert(value, 25, 7) }; get() = fcr31.extract(25, 7)

    val registers = CpuRegisters()
    var r0: Int; set(value) = run { registers.r0 = value }; get() = registers.r0
    var r1: Int; set(value) = run { registers.r1 = value }; get() = registers.r1
    var r2: Int; set(value) = run { registers.r2 = value }; get() = registers.r2
    var r3: Int; set(value) = run { registers.r3 = value }; get() = registers.r3
    var r4: Int; set(value) = run { registers.r4 = value }; get() = registers.r4
    var r5: Int; set(value) = run { registers.r5 = value }; get() = registers.r5
    var r6: Int; set(value) = run { registers.r6 = value }; get() = registers.r6
    var r7: Int; set(value) = run { registers.r7 = value }; get() = registers.r7
    var r8: Int; set(value) = run { registers.r8 = value }; get() = registers.r8
    var r9: Int; set(value) = run { registers.r9 = value }; get() = registers.r9
    var r10: Int; set(value) = run { registers.r10 = value }; get() = registers.r10
    var r11: Int; set(value) = run { registers.r11 = value }; get() = registers.r11
    var r12: Int; set(value) = run { registers.r12 = value }; get() = registers.r12
    var r13: Int; set(value) = run { registers.r13 = value }; get() = registers.r13
    var r14: Int; set(value) = run { registers.r14 = value }; get() = registers.r14
    var r15: Int; set(value) = run { registers.r15 = value }; get() = registers.r15
    var r16: Int; set(value) = run { registers.r16 = value }; get() = registers.r16
    var r17: Int; set(value) = run { registers.r17 = value }; get() = registers.r17
    var r18: Int; set(value) = run { registers.r18 = value }; get() = registers.r18
    var r19: Int; set(value) = run { registers.r19 = value }; get() = registers.r19
    var r20: Int; set(value) = run { registers.r20 = value }; get() = registers.r20
    var r21: Int; set(value) = run { registers.r21 = value }; get() = registers.r21
    var r22: Int; set(value) = run { registers.r22 = value }; get() = registers.r22
    var r23: Int; set(value) = run { registers.r23 = value }; get() = registers.r23
    var r24: Int; set(value) = run { registers.r24 = value }; get() = registers.r24
    var r25: Int; set(value) = run { registers.r25 = value }; get() = registers.r25
    var r26: Int; set(value) = run { registers.r26 = value }; get() = registers.r26
    var r27: Int; set(value) = run { registers.r27 = value }; get() = registers.r27
    var r28: Int; set(value) = run { registers.r28 = value }; get() = registers.r28
    var r29: Int; set(value) = run { registers.r29 = value }; get() = registers.r29
    var r30: Int; set(value) = run { registers.r30 = value }; get() = registers.r30
    var r31: Int; set(value) = run { registers.r31 = value }; get() = registers.r31
    fun getGpr(index: Int): Int = registers.getGpr(index)
    fun setGpr(index: Int, v: Int): Unit = registers.setGpr(index, v)

    fun writeRegisters(addr: Int, start: Int = 0, count: Int = 32 - start) {
        for (n in 0 until count) mem.sw(addr + n * 4, getGpr(start + n))
    }

    fun readRegisters(addr: Int, start: Int = 0, count: Int = 32 - start) {
        for (n in 0 until count) setGpr(start + n, mem.lw(addr + n * 4))
    }

    //val FPR = FloatArray(32) { 0f }
    //val FPR_I = FprI(this)

    var IR: Int; set(value) = run { registers.IR = InstructionRegister(value) }; get() = registers.IR.data
    var _PC: Int; set(value) = run { registers.PC = value }; get() = registers.PC
    var _nPC: Int; set(value) = run { registers.nPC = value }; get() = registers.nPC
    var LO: Int; set(value) = run { registers.LO = value }; get() = registers.LO
    var HI: Int; set(value) = run { registers.HI = value }; get() = registers.HI
    var IC: Int; set(value) = run { registers.IC = value }; get() = registers.IC

    fun getPCRef() = ::sPC

    var sPC get() = PC; set(value) = run { setPC(value) }

    val PC: Int get() = _PC

    var HI_LO: Long
        get() = (HI.toLong() shl 32) or (LO.toLong() and 0xFFFFFFFF)
        set(value) {
            HI = (value ushr 32).toInt()
            LO = (value ushr 0).toInt()
        }

    fun setPC(pc: Int) {
        _PC = pc
        _nPC = pc + 4
    }

    inline fun jump(pc: Int) {
        _PC = pc
        _nPC = pc + 4
    }

    inline fun advance_pc(offset: Int) {
        _PC = _nPC
        _nPC += offset
    }

    //fun getGpr(index: Int): Int = _R[index and 0x1F]
    //fun setGpr(index: Int, v: Int): Unit = run { if (index != 0) _R[index and 0x1F] = v }

    fun getFpr(index: Int): Float = _F[index]
    fun setFpr(index: Int, v: Float): Unit = run { _F[index] = v }

    fun getFprI(index: Int): Int = _FI[index]
    fun setFprI(index: Int, v: Int): Unit = run { _FI[index] = v }

    fun setVfpr(index: Int, value: Float) = run { _VFPR[index] = value }
    fun getVfpr(index: Int): Float = _VFPR[index]

    val VFPR = VFPRF_Class()
    val VFPRI = VFPRI_Class()

    fun setVfprI(index: Int, value: Int) = run { _VFPR_I[index] = value }
    fun getVfprI(index: Int): Int = _VFPR_I[index]

    fun syscall(syscall: Int): Unit = syscalls.syscall(this, syscall)

    fun _break(syscall: Int): Unit = throw CpuBreakExceptionCached(syscall)

    inner class VFPRI_Class {
        operator fun get(index: Int): Int = getVfprI(index)
        operator fun set(index: Int, value: Int): Unit = run { setVfprI(index, value) }
    }

    inner class VFPRF_Class {
        operator fun get(index: Int): Float = getVfpr(index)
        operator fun set(index: Int, value: Float): Unit = run { setVfpr(index, value) }
    }

    fun clone() = CpuState("${this.name}.cloned", globalCpuState, syscalls).apply {
        this@CpuState.copyTo(this)
    }

    fun setTo(src: CpuState) = run { src.copyTo(this) }

    fun copyTo(dst: CpuState) {
        val src = this
        dst._PC = src._PC
        dst._nPC = src._nPC
        dst.HI = src.HI
        dst.LO = src.LO
        dst.IC = src.IC
        dst.IR = src.IR
        dst.fcr0 = src.fcr0
        dst.fcr25 = src.fcr25
        dst.fcr26 = src.fcr26
        dst.fcr27 = src.fcr27
        dst.fcr28 = src.fcr28
        dst.fcr31 = src.fcr31
        for (n in 0 until 32) dst.setGpr(n, src.getGpr(n))
        for (n in 0 until 32) dst.setFpr(n, src.getFpr(n))
        for (n in 0 until 128) dst.setVfpr(n, src.getVfpr(n))
    }

    val summary: String
        //get() = "REGS($id)[" + (0 until 32).map { "r%d=%d".format(it, getGpr(it)) }.joinToString(", ") + "]"
        get() = "REGS($name:$id)[" + (0 until 32).map {
            "r%d=0x%08X".format(
                it,
                getGpr(it)
            )
        }.joinToString(", ") + "][PC = ${PC.hex}]"

    fun dump() {
        println(" DUMP:-- $summary")
    }

    object VFPU_CTRL {
        const val SPREFIX = 0
        const val TPREFIX = 1
        const val DPREFIX = 2
        const val CC = 3
        const val INF4 = 4
        const val RSV5 = 5
        const val RSV6 = 6
        const val REV = 7
        const val RCX0 = 8
        const val RCX1 = 9
        const val RCX2 = 10
        const val RCX3 = 11
        const val RCX4 = 12
        const val RCX5 = 13
        const val RCX6 = 14
        const val RCX7 = 15
        const val MAX = 16
    }

    object VCondition {
        const val FL = 0
        const val EQ = 1
        const val LT = 2
        const val LE = 3
        const val TR = 4
        const val NE = 5
        const val GE = 6
        const val GT = 7
        const val EZ = 8
        const val EN = 9
        const val EI = 10
        const val ES = 11
        const val NZ = 12
        const val NN = 13
        const val NI = 14
        const val NS = 15
    }

    fun div(RS: Int, RT: Int) = run { LO = RS / RT; HI = RS % RT }
    fun divu(RS: Int, RT: Int) {
        val d = RT
        if (d != 0) {
            LO = RS udiv d
            HI = RS urem d
        } else {
            LO = 0
            HI = 0
        }
    }

    private val itemp = IntArray(2)
    fun mult(RS: Int, RT: Int) =
        run { imul32_64(RS, RT, itemp); this.LO = itemp[0]; this.HI = itemp[1] }

    fun multu(RS: Int, RT: Int) =
        run { umul32_64(RS, RT, itemp); this.LO = itemp[0]; this.HI = itemp[1] }

    fun madd(RS: Int, RT: Int) = run { HI_LO += RS.toLong() * RT.toLong() }
    fun maddu(RS: Int, RT: Int) = run { HI_LO += RS.unsigned * RT.unsigned }
    fun msub(RS: Int, RT: Int) = run { HI_LO -= RS.toLong() * RT.toLong() }
    fun msubu(RS: Int, RT: Int) = run { HI_LO -= RS.unsigned * RT.unsigned }

    fun lb(addr: Int) = mem.lb(addr)
    fun lbu(addr: Int) = mem.lbu(addr)
    fun lh(addr: Int) = mem.lh(addr)
    fun lhu(addr: Int) = mem.lhu(addr)
    fun lw(addr: Int) = mem.lw(addr)

    fun lwl(addr: Int, value: Int) = mem.lwl(addr, value)
    fun lwr(addr: Int, value: Int) = mem.lwr(addr, value)
    fun swl(addr: Int, value: Int) = mem.swl(addr, value)
    fun swr(addr: Int, value: Int) = mem.swr(addr, value)

    fun sb(addr: Int, value: Int) = mem.sb(addr, value)
    fun sh(addr: Int, value: Int) = mem.sh(addr, value)
    fun sw(addr: Int, value: Int) = mem.sw(addr, value)

    fun slt(RS: Int, RT: Int): Int = (RS < RT).toInt()
    fun sltu(RS: Int, RT: Int): Int = (RS ult RT).toInt()

    fun _checkFNan(FD: Float) {
        if (FD.isNaN()) fcr31 = fcr31 or 0x00010040
        if (FD.isInfinite()) fcr31 = fcr31 or 0x00005014
    }

    fun fmov(RS: Float): Float = RS
    fun fadd(RS: Float, RT: Float): Float = RS pspAdd RT
    fun fsub(RS: Float, RT: Float): Float = RS pspSub RT
    fun fmul(RS: Float, RT: Float): Float {
        val res = RS * RT
        return if (fcr31_fs && res.isAlmostZero1()) 0f else res
    }

    fun fdiv(RS: Float, RT: Float): Float = RS / RT
    fun fneg(v: Float): Float = -v
    fun fabs(v: Float): Float = kotlin.math.abs(v)
    fun fsqrt(v: Float): Float = kotlin.math.sqrt(v)

    fun cvt_s_w(v: Int): Float = v.toFloat()
    fun cvt_w_s(FS: Float): Int {
        return when (fcr31_rm) {
            0 -> Math.rint(FS) // rint: round nearest
            1 -> Math.cast(FS) // round to zero
            2 -> Math.ceil(FS) // round up (ceil)
            3 -> Math.floor(FS) // round down (floor)
            else -> FS.toInt()
        }
    }

    fun cfc1(IR_rd: Int, RT: Int): Int {
        return when (IR_rd) {
            0 -> fcr0
            25 -> fcr25
            26 -> fcr26
            27 -> fcr27
            28 -> fcr28
            31 -> fcr31
            else -> -1
        }
    }

    fun ctc1(IR_rd: Int, RT: Int) {
        when (IR_rd) {
            31 -> updateFCR31(RT)
        }
    }

    private inline fun _cu(FS: Float, FT: Float, callback: () -> Boolean): Boolean =
        if (FS.isNaN() || FT.isNaN()) true else callback()

    private inline fun _co(FS: Float, FT: Float, callback: () -> Boolean): Boolean =
        if (FS.isNaN() || FT.isNaN()) false else callback()

    fun c_f_s(FS: Float, FT: Float) = _co(FS, FT) { false }
    fun c_un_s(FS: Float, FT: Float) = _cu(FS, FT) { false }
    fun c_eq_s(FS: Float, FT: Float) = _co(FS, FT) { FS == FT }
    fun c_ueq_s(FS: Float, FT: Float) = _cu(FS, FT) { FS == FT }
    fun c_olt_s(FS: Float, FT: Float) = _co(FS, FT) { FS < FT }
    fun c_ult_s(FS: Float, FT: Float) = _cu(FS, FT) { FS < FT }
    fun c_ole_s(FS: Float, FT: Float) = _co(FS, FT) { FS <= FT }
    fun c_ule_s(FS: Float, FT: Float) = _cu(FS, FT) { FS <= FT }

    fun f_get_fcr31_cc() = fcr31_cc
    fun f_get_fcr31_cc_not() = !fcr31_cc

    //fun syscall(id: Int) = mem.sw(addr, value)
}

open class VfpuPrefix(val defaultValue: Int, val array: IntArray, val arrayIndex: Int) {
    var info get() = array[arrayIndex]; set(value) = run { array[arrayIndex] = value }
    var enabled = false

    fun setEnable(v: Int) {
        info = v
        enabled = true
    }

    fun consume() {
        enabled = false
    }
}

class VfpuSourceTargetPrefix(array: IntArray, arrayIndex: Int) : VfpuPrefix(0xDC0000E4.toInt(), array, arrayIndex) {
    private val temp = Float32BufferAlloc(16)

    fun applyAndConsume(inout: Float32Buffer, size: Int = inout.size) {
        if (!enabled) return
        apply(inout, size)
        enabled = false
    }

    fun apply(inout: Float32Buffer, size: Int = inout.size) {
        if (!enabled) return
        arraycopy(inout, 0, temp, 0, size)
        for (n in 0 until size) {
            val sourceIndex = (info ushr (0 + n * 2)) and 3
            val sourceAbsolute = ((info ushr (8 + n * 1)) and 1) != 0
            val sourceConstant = ((info ushr (12 + n * 1)) and 1) != 0
            val sourceNegate = ((info ushr (16 + n * 1)) and 1) != 0

            val value: Float = if (sourceConstant) {
                when (sourceIndex) {
                    0 -> if (sourceAbsolute) 3f else 0f
                    1 -> if (sourceAbsolute) 1f / 3f else 1f
                    2 -> if (sourceAbsolute) 1f / 4f else 2f
                    3 -> if (sourceAbsolute) 1f / 6f else 1f / 2f
                    else -> invalidOp1
                }
            } else {
                if (sourceAbsolute) temp[sourceIndex].pspAbs else temp[sourceIndex]
            }

            inout[n] = if (sourceNegate) -value else value
        }
    }
}

class VfpuDestinationPrefix(array: IntArray, arrayIndex: Int) : VfpuPrefix(0xDC0000E4.toInt(), array, arrayIndex) {
    private fun mask(n: Int) = ((info ushr (8 + n * 1)) and 1) != 0
    fun saturation(n: Int) = (info ushr (0 + n * 2)) and 3

    fun mustWrite(n: Int) = if (enabled) !mask(n) else true

    fun transform(n: Int, value: Float): Float {
        if (!enabled) return value
        return when (saturation(n)) {
            1 -> value.pspSat0
            3 -> value.pspSat1
            else -> value
        }
    }
}

/*
class CpuState(val mem: Memory, val syscalls: Syscalls = TraceSyscallHandler()) {
	var r0: Int; set(value) = Unit; get() = 0
	var r1: Int = 0
	var r2: Int = 0
	var r3: Int = 0
	var r4: Int = 0
	var r5: Int = 0
	var r6: Int = 0
	var r7: Int = 0
	var r8: Int = 0
	var r9: Int = 0
	var r10: Int = 0
	var r11: Int = 0
	var r12: Int = 0
	var r13: Int = 0
	var r14: Int = 0
	var r15: Int = 0
	var r16: Int = 0
	var r17: Int = 0
	var r18: Int = 0
	var r19: Int = 0
	var r20: Int = 0
	var r21: Int = 0
	var r22: Int = 0
	var r23: Int = 0
	var r24: Int = 0
	var r25: Int = 0
	var r26: Int = 0
	var r27: Int = 0
	var r28: Int = 0
	var r29: Int = 0
	var r30: Int = 0
	var r31: Int = 0

	val GPR = Gpr(this)

	var IR: Int = 0
	var _PC: Int = 0
	var _nPC: Int = 0
	var LO: Int = 0
	var HI: Int = 0
	var IC: Int = 0

	fun setPC(pc: Int) {
		_PC = pc
		_nPC = pc + 4
	}

	fun getPC() = _PC

	fun jump(pc: Int) {
		_PC = pc
		_nPC = pc + 4
	}

	fun advance_pc(offset: Int) {
		_PC = _nPC
		_nPC += offset
	}

	fun getGpr(index: Int): Int = GPR[index]
	fun setGpr(index: Int, v: Int): Unit {
		GPR[index] = value
	}

	class Gpr(val state: CpuState) {
		// ERROR!

		//fun ref(index: Int): KMutableProperty<Int> = state.run {
		//	when (index) {
		//		0 -> ::r0; 1 -> ::r1; 2 -> ::r2; 3 -> ::r3;
		//		4 -> ::r4; 5 -> ::r5; 6 -> ::r6; 7 -> ::r7;
		//		8 -> ::r8; 9 -> ::r9; 10 -> ::r10; 11 -> ::r11;
		//		12 -> ::r12; 13 -> ::r13; 14 -> ::r14; 15 -> ::r15;
		//		16 -> ::r16; 17 -> ::r17; 18 -> ::r18; 19 -> ::r19;
		//		20 -> ::r20; 21 -> ::r21; 22 -> ::r22; 23 -> ::r23;
		//		24 -> ::r24; 25 -> ::r25; 26 -> ::r26; 27 -> ::r27;
		//		28 -> ::r28; 29 -> ::r29; 30 -> ::r30; 31 -> ::r31
		//		else -> ::r0
		//	}
		//}

		fun hex(index: Int): String = "0x%08X".format(get(index))

		operator fun get(index: Int): Int = state.run {
			when (index and 0x1F) {
				0 -> r0; 1 -> r1; 2 -> r2; 3 -> r3
				4 -> r4; 5 -> r5; 6 -> r6; 7 -> r7
				8 -> r8; 9 -> r9; 10 -> r10; 11 -> r11
				12 -> r12; 13 -> r13; 14 -> r14; 15 -> r15
				16 -> r16; 17 -> r17; 18 -> r18; 19 -> r19
				20 -> r20; 21 -> r21; 22 -> r22; 23 -> r23
				24 -> r24; 25 -> r25; 26 -> r26; 27 -> r27
				28 -> r28; 29 -> r29; 30 -> r30; 31 -> r31
				else -> 0
			}
		}

		operator fun set(index: Int, v: Int): Unit = state.run {
			when (index and 0x1F) {
				0 -> r0 = v; 1 -> r1 = v; 2 -> r2 = v; 3 -> r3 = v
				4 -> r4 = v; 5 -> r5 = v; 6 -> r6 = v; 7 -> r7 = v
				8 -> r8 = v; 9 -> r9 = v; 10 -> r10 = v; 11 -> r11 = v
				12 -> r12 = v; 13 -> r13 = v; 14 -> r14 = v; 15 -> r15 = v
				16 -> r16 = v; 17 -> r17 = v; 18 -> r18 = v; 19 -> r19 = v
				20 -> r20 = v; 21 -> r21 = v; 22 -> r22 = v; 23 -> r23 = v
				24 -> r24 = v; 25 -> r25 = v; 26 -> r26 = v; 27 -> r27 = v
				28 -> r28 = v; 29 -> r29 = v; 30 -> r30 = v; 31 -> r31 = v
				else -> Unit
			}
		}
	}

	fun syscall(syscall: Int): Unit = syscalls.syscall(this, syscall)
}
*/

enum class VfpuSingleRegisters {
    S000, S010, S020, S030, S100, S110, S120, S130,
    S200, S210, S220, S230, S300, S310, S320, S330,
    S400, S410, S420, S430, S500, S510, S520, S530,
    S600, S610, S620, S630, S700, S710, S720, S730,
    S001, S011, S021, S031, S101, S111, S121, S131,
    S201, S211, S221, S231, S301, S311, S321, S331,
    S401, S411, S421, S431, S501, S511, S521, S531,
    S601, S611, S621, S631, S701, S711, S721, S731,
    S002, S012, S022, S032, S102, S112, S122, S132,
    S202, S212, S222, S232, S302, S312, S322, S332,
    S402, S412, S422, S432, S502, S512, S522, S532,
    S602, S612, S622, S632, S702, S712, S722, S732,
    S003, S013, S023, S033, S103, S113, S123, S133,
    S203, S213, S223, S233, S303, S313, S323, S333,
    S403, S413, S423, S433, S503, S513, S523, S533,
    S603, S613, S623, S633, S703, S713, S723, S733;

    companion object {
        val values = values()
    }
}

enum class VfpuPairRegisters {
    C000, C010, C020, C030, C100, C110, C120, C130,
    C200, C210, C220, C230, C300, C310, C320, C330,
    C400, C410, C420, C430, C500, C510, C520, C530,
    C600, C610, C620, C630, C700, C710, C720, C730,
    R000, R001, R002, R003, R100, R101, R102, R103,
    R200, R201, R202, R203, R300, R301, R302, R303,
    R400, R401, R402, R403, R500, R501, R502, R503,
    R600, R601, R602, R603, R700, R701, R702, R703,
    C002, C012, C022, C032, C102, C112, C122, C132,
    C202, C212, C222, C232, C302, C312, C322, C332,
    C402, C412, C422, C432, C502, C512, C522, C532,
    C602, C612, C622, C632, C702, C712, C722, C732,
    R020, R021, R022, R023, R120, R121, R122, R123,
    R220, R221, R222, R223, R320, R321, R322, R323,
    R420, R421, R422, R423, R520, R521, R522, R523,
    R620, R621, R622, R623, R720, R721, R722, R723;

    companion object {
        val values = values()
    }
}

enum class VfpuTripletRegisters {
    C000, C010, C020, C030, C100, C110, C120, C130,
    C200, C210, C220, C230, C300, C310, C320, C330,
    C400, C410, C420, C430, C500, C510, C520, C530,
    C600, C610, C620, C630, C700, C710, C720, C730,
    R000, R001, R002, R003, R100, R101, R102, R103,
    R200, R201, R202, R203, R300, R301, R302, R303,
    R400, R401, R402, R403, R500, R501, R502, R503,
    R600, R601, R602, R603, R700, R701, R702, R703,
    C001, C011, C021, C031, C101, C111, C121, C131,
    C201, C211, C221, C231, C301, C311, C321, C331,
    C401, C411, C421, C431, C501, C511, C521, C531,
    C601, C611, C621, C631, C701, C711, C721, C731,
    R010, R011, R012, R013, R110, R111, R112, R113,
    R210, R211, R212, R213, R310, R311, R312, R313,
    R410, R411, R412, R413, R510, R511, R512, R513,
    R610, R611, R612, R613, R710, R711, R712, R713;

    companion object {
        val values = VfpuPairRegisters.values()
    }
}

/*
+static const char * const vfpu_vqreg_names[128] = {
+  "C000",  "C010",  "C020",  "C030",  "C100",  "C110",  "C120",  "C130",
+  "C200",  "C210",  "C220",  "C230",  "C300",  "C310",  "C320",  "C330",
+  "C400",  "C410",  "C420",  "C430",  "C500",  "C510",  "C520",  "C530",
+  "C600",  "C610",  "C620",  "C630",  "C700",  "C710",  "C720",  "C730",
+  "R000",  "R001",  "R002",  "R003",  "R100",  "R101",  "R102",  "R103",
+  "R200",  "R201",  "R202",  "R203",  "R300",  "R301",  "R302",  "R303",
+  "R400",  "R401",  "R402",  "R403",  "R500",  "R501",  "R502",  "R503",
+  "R600",  "R601",  "R602",  "R603",  "R700",  "R701",  "R702",  "R703",
+  "",  "",  "",  "",  "",  "",  "",  "",
+  "",  "",  "",  "",  "",  "",  "",  "",
+  "",  "",  "",  "",  "",  "",  "",  "",
+  "",  "",  "",  "",  "",  "",  "",  "",
+  "",  "",  "",  "",  "",  "",  "",  "",
+  "",  "",  "",  "",  "",  "",  "",  "",
+  "",  "",  "",  "",  "",  "",  "",  "",
+  "",  "",  "",  "",  "",  "",  "",  ""
+};
+
+static const char * const vfpu_mpreg_names[128] = {
+  "M000",  "",  "M020",  "",  "M100",  "",  "M120",  "",
+  "M200",  "",  "M220",  "",  "M300",  "",  "M320",  "",
+  "M400",  "",  "M420",  "",  "M500",  "",  "M520",  "",
+  "M600",  "",  "M620",  "",  "M700",  "",  "M720",  "",
+  "E000",  "",  "E002",  "",  "E100",  "",  "E102",  "",
+  "E200",  "",  "E202",  "",  "E300",  "",  "E302",  "",
+  "E400",  "",  "E402",  "",  "E500",  "",  "E502",  "",
+  "E600",  "",  "E602",  "",  "E700",  "",  "E702",  "",
+  "M002",  "",  "M022",  "",  "M102",  "",  "M122",  "",
+  "M202",  "",  "M222",  "",  "M302",  "",  "M322",  "",
+  "M402",  "",  "M422",  "",  "M502",  "",  "M522",  "",
+  "M602",  "",  "M622",  "",  "M702",  "",  "M722",  "",
+  "E020",  "",  "E022",  "",  "E120",  "",  "E122",  "",
+  "E220",  "",  "E222",  "",  "E320",  "",  "E322",  "",
+  "E420",  "",  "E422",  "",  "E520",  "",  "E522",  "",
+  "E620",  "",  "E622",  "",  "E720",  "",  "E722",  ""
+};
+
+static const char * const vfpu_mtreg_names[128] = {
+  "M000",  "M010",  "",  "",  "M100",  "M110",  "",  "",
+  "M200",  "M210",  "",  "",  "M300",  "M310",  "",  "",
+  "M400",  "M410",  "",  "",  "M500",  "M510",  "",  "",
+  "M600",  "M610",  "",  "",  "M700",  "M710",  "",  "",
+  "E000",  "E001",  "",  "",  "E100",  "E101",  "",  "",
+  "E200",  "E201",  "",  "",  "E300",  "E301",  "",  "",
+  "E400",  "E401",  "",  "",  "E500",  "E501",  "",  "",
+  "E600",  "E601",  "",  "",  "E700",  "E701",  "",  "",
+  "M001",  "M011",  "",  "",  "M101",  "M111",  "",  "",
+  "M201",  "M211",  "",  "",  "M301",  "M311",  "",  "",
+  "M401",  "M411",  "",  "",  "M501",  "M511",  "",  "",
+  "M601",  "M611",  "",  "",  "M701",  "M711",  "",  "",
+  "E010",  "E011",  "",  "",  "E110",  "E111",  "",  "",
+  "E210",  "E211",  "",  "",  "E310",  "E311",  "",  "",
+  "E410",  "E411",  "",  "",  "E510",  "E511",  "",  "",
+  "E610",  "E611",  "",  "",  "E710",  "E711",  "",  ""
+};
+
+static const char * const vfpu_mqreg_names[128] = {
+  "M000",  "",  "",  "",  "M100",  "",  "",  "",
+  "M200",  "",  "",  "",  "M300",  "",  "",  "",
+  "M400",  "",  "",  "",  "M500",  "",  "",  "",
+  "M600",  "",  "",  "",  "M700",  "",  "",  "",
+  "E000",  "",  "",  "",  "E100",  "",  "",  "",
+  "E200",  "",  "",  "",  "E300",  "",  "",  "",
+  "E400",  "",  "",  "",  "E500",  "",  "",  "",
+  "E600",  "",  "",  "",  "E700",  "",  "",  "",
+  "",  "",  "",  "",  "",  "",  "",  "",
+  "",  "",  "",  "",  "",  "",  "",  "",
+  "",  "",  "",  "",  "",  "",  "",  "",
+  "",  "",  "",  "",  "",  "",  "",  "",
+  "",  "",  "",  "",  "",  "",  "",  "",
+  "",  "",  "",  "",  "",  "",  "",  "",
+  "",  "",  "",  "",  "",  "",  "",  "",
+  "",  "",  "",  "",  "",  "",  "",  ""
+};
 */