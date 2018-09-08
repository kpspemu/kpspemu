package com.soywiz.kpspemu.cpu.interpreter

import com.soywiz.klogger.*
import com.soywiz.kmem.*
import com.soywiz.korim.color.*
import com.soywiz.korio.crypto.*
import com.soywiz.korio.error.*
import com.soywiz.korio.lang.*
import com.soywiz.korma.math.*
import com.soywiz.kpspemu.cpu.*
import com.soywiz.kpspemu.cpu.CpuState.*
import com.soywiz.kpspemu.cpu.dis.*
import com.soywiz.kpspemu.hle.manager.*
import com.soywiz.kpspemu.mem.*
import com.soywiz.kpspemu.util.*
import kotlin.math.*

class CpuInterpreter(
    var cpu: CpuState,
    val breakpoints: Breakpoints,
    val nameProvider: NameProvider,
    var trace: Boolean = false
) {
    val interpreter = InstructionInterpreter()
    val dispatcher = InstructionInterpreterDispatcher()

    fun steps(count: Int, trace: Boolean = false): Int = stepsNormal(count, trace)

    fun stepsNormal(count: Int, trace: Boolean): Int {
        val cpu = this.cpu
        val regs = cpu.regs
        val mem = cpu.mem
        var sPC = 0
        var n = 0
        val dispatcher = this.dispatcher
        //val fast = (mem as FastMemory).buffer
        val breakpointsEnabled = breakpoints.enabled
        val interpreter = interpreter
        try {
            val mustCheck = trace || breakpoints.enabled
            while (n < count) {
                sPC = regs.PC
                if (mustCheck) {
                    checkTrace(sPC, cpu)
                }
                n++
                //if (PC == 0) throw IllegalStateException("Trying to execute PC=0")
                if (trace) tracePC()
                //println("%08X".format(sPC))
                val IR = mem.lw(sPC)
                regs.IR = InstructionRegister(IR)
                dispatcher.dispatch(interpreter, cpu, regs, mem, sPC, IR)
            }
        } catch (e: Throwable) {
            checkException(sPC, e)
        } finally {
            cpu.totalExecuted += n
        }
        return n
    }

    private fun checkTrace(sPC: Int, cpu: CpuState) {
        if (trace) doTrace(sPC, cpu)
        if (breakpoints.enabled && breakpoints[sPC]) throw BreakpointException(cpu, sPC)
    }

    private fun doTrace(sPC: Int, state: CpuState) {
        val I = if (state.globalCpuState.insideInterrupt) "I" else "_"
        println("TRACE[$I][${state._thread?.name}]:${sPC.hex} : ${cpu.mem.disasmMacro(sPC, nameProvider)}")
    }

    private fun checkException(sPC: Int, e: Throwable) {
        if (e !is EmulatorControlFlowException) {
            Console.error("There was an error at 0x%08X: %s".format(sPC, cpu.mem.disasmMacro(sPC, nameProvider)))
            Console.error(" - RA at 0x%08X: %s".format(cpu.RA, cpu.mem.disasmMacro(cpu.RA, nameProvider)))
        }
        throw e
    }


    private fun tracePC() {
        println("0x%08X: %s".format(cpu._PC, cpu.mem.disasmMacro(cpu._PC, nameProvider)))
    }
}


@Suppress("FunctionName", "UNUSED_PARAMETER", "unused")
// http://www.mrc.uidaho.edu/mrc/people/jff/digital/MIPSir.html
//class InstructionInterpreter {
//class InstructionInterpreter : InstructionEvaluator<CpuState>() {
class InstructionInterpreter : InstructionDecoder() {
    val VDEST2 = IntArray2(4, 4)
    val VSRC = IntArray(16)
    val mc = MatrixContext()
    val tempRegs = IntArray(16)
    val tempRegs2 = IntArray2(4, 4)

    inline fun unimplemented(s: CpuState, i: InstructionType): Unit =
        TODO("unimplemented: ${i.name} : " + i + " at ${"%08X".format(s._PC)}")

    // ALU
    inline fun lui(s: CpuState, r: CpuRegisters, m: Memory) = r { RT = (U_IMM16 shl 16) }

    inline fun movz(s: CpuState, r: CpuRegisters, m: Memory) = r { RD = s.movz(RT, RD, RS) }
    inline fun movn(s: CpuState, r: CpuRegisters, m: Memory) = r { RD = s.movn(RT, RD, RS) }

    inline fun ext(s: CpuState, r: CpuRegisters, m: Memory) = r { RT = s.ext(RS, POS, SIZE_E) }
    inline fun ins(s: CpuState, r: CpuRegisters, m: Memory) = r { RT = s.ins(RT, RS, POS, SIZE_I) }

    inline fun clz(s: CpuState, r: CpuRegisters, m: Memory) = r { RD = s.clz(RS) }
    inline fun clo(s: CpuState, r: CpuRegisters, m: Memory) = r { RD = s.clo(RS) }
    inline fun seb(s: CpuState, r: CpuRegisters, m: Memory) = r { RD = s.seb(RT) }
    inline fun seh(s: CpuState, r: CpuRegisters, m: Memory) = r { RD = s.seh(RT) }

    inline fun wsbh(s: CpuState, r: CpuRegisters, m: Memory) = r { RD = s.wsbh(RT) }
    inline fun wsbw(s: CpuState, r: CpuRegisters, m: Memory) = r { RD = s.wsbw(RT) }

    inline fun max(s: CpuState, r: CpuRegisters, m: Memory) = r { RD = s.max(RS, RT) }
    inline fun min(s: CpuState, r: CpuRegisters, m: Memory) = r { RD = s.min(RS, RT) }

    inline fun add(s: CpuState, r: CpuRegisters, m: Memory) = r { RD = RS + RT }
    inline fun addu(s: CpuState, r: CpuRegisters, m: Memory) = r { RD = RS + RT }
    inline fun sub(s: CpuState, r: CpuRegisters, m: Memory) = r { RD = RS - RT }
    inline fun subu(s: CpuState, r: CpuRegisters, m: Memory) = r { RD = RS - RT }
    inline fun addi(s: CpuState, r: CpuRegisters, m: Memory) = r { RT = RS + S_IMM16 }
    inline fun addiu(s: CpuState, r: CpuRegisters, m: Memory) = r { RT = RS + S_IMM16 }

    inline fun div(s: CpuState, r: CpuRegisters, m: Memory) = r { s.div(RS, RT) }
    inline fun divu(s: CpuState, r: CpuRegisters, m: Memory) = r { s.divu(RS, RT) }
    inline fun mult(s: CpuState, r: CpuRegisters, m: Memory) = r { s.mult(RS, RT) }
    inline fun multu(s: CpuState, r: CpuRegisters, m: Memory) = r { s.multu(RS, RT) }
    inline fun madd(s: CpuState, r: CpuRegisters, m: Memory) = r { s.madd(RS, RT) }
    inline fun maddu(s: CpuState, r: CpuRegisters, m: Memory) = r { s.maddu(RS, RT) }
    inline fun msub(s: CpuState, r: CpuRegisters, m: Memory) = r { s.msub(RS, RT) }
    inline fun msubu(s: CpuState, r: CpuRegisters, m: Memory) = r { s.msubu(RS, RT) }

    inline fun mflo(s: CpuState, r: CpuRegisters, m: Memory) = r { RD = LO }
    inline fun mfhi(s: CpuState, r: CpuRegisters, m: Memory) = r { RD = HI }
    inline fun mfic(s: CpuState, r: CpuRegisters, m: Memory) = r { RT = IC }

    inline fun mtlo(s: CpuState, r: CpuRegisters, m: Memory) = r { LO = RS }
    inline fun mthi(s: CpuState, r: CpuRegisters, m: Memory) = r { HI = RS }
    inline fun mtic(s: CpuState, r: CpuRegisters, m: Memory) = r { IC = RT }

    // ALU: Bit
    inline fun or(s: CpuState, r: CpuRegisters, m: Memory) = r { RD = RS or RT }
    inline fun xor(s: CpuState, r: CpuRegisters, m: Memory) = r { RD = RS xor RT }
    inline fun and(s: CpuState, r: CpuRegisters, m: Memory) = r { RD = RS and RT }
    inline fun nor(s: CpuState, r: CpuRegisters, m: Memory) = r { RD = s.nor(RS, RT) }

    inline fun ori(s: CpuState, r: CpuRegisters, m: Memory) = r { RT = RS or U_IMM16 }
    inline fun xori(s: CpuState, r: CpuRegisters, m: Memory) = r { RT = RS xor U_IMM16 }
    inline fun andi(s: CpuState, r: CpuRegisters, m: Memory) = r { RT = RS and U_IMM16 }

    inline fun sll(s: CpuState, r: CpuRegisters, m: Memory) = r { RD = s.sll(RT, POS) }
    inline fun sra(s: CpuState, r: CpuRegisters, m: Memory) = r { RD = s.sra(RT, POS) }
    inline fun srl(s: CpuState, r: CpuRegisters, m: Memory) = r { RD = s.srl(RT, POS) }
    inline fun sllv(s: CpuState, r: CpuRegisters, m: Memory) = r { RD = s.sll(RT, RS) }
    inline fun srav(s: CpuState, r: CpuRegisters, m: Memory) = r { RD = s.sra(RT, RS) }
    inline fun srlv(s: CpuState, r: CpuRegisters, m: Memory) = r { RD = s.srl(RT, RS) }

    inline fun bitrev(s: CpuState, r: CpuRegisters, m: Memory) = r { RD = s.bitrev32(RT) }

    inline fun rotr(s: CpuState, r: CpuRegisters, m: Memory) = r { RD = s.rotr(RT, POS) }
    inline fun rotrv(s: CpuState, r: CpuRegisters, m: Memory) = r { RD = s.rotr(RT, RS) }

    // Memory
    inline fun lb(s: CpuState, r: CpuRegisters, m: Memory) = r { RT = m.lb(RS_IMM16) }

    inline fun lbu(s: CpuState, r: CpuRegisters, m: Memory) = r { RT = m.lbu(RS_IMM16) }
    inline fun lh(s: CpuState, r: CpuRegisters, m: Memory) = r { RT = m.lh(RS_IMM16) }
    inline fun lhu(s: CpuState, r: CpuRegisters, m: Memory) = r { RT = m.lhu(RS_IMM16) }
    inline fun lw(s: CpuState, r: CpuRegisters, m: Memory) = r { RT = m.lw(RS_IMM16) }
    inline fun ll(s: CpuState, r: CpuRegisters, m: Memory) = r { RT = m.lw(RS_IMM16) }

    inline fun lwl(s: CpuState, r: CpuRegisters, m: Memory) = r { RT = m.lwl(RS_IMM16, RT) }
    inline fun lwr(s: CpuState, r: CpuRegisters, m: Memory) = r { RT = m.lwr(RS_IMM16, RT) }

    inline fun swl(s: CpuState, r: CpuRegisters, m: Memory) = r { m.swl(RS_IMM16, RT) }
    inline fun swr(s: CpuState, r: CpuRegisters, m: Memory) = r { m.swr(RS_IMM16, RT) }

    inline fun sb(s: CpuState, r: CpuRegisters, m: Memory) = r { m.sb(RS_IMM16, RT) }
    inline fun sh(s: CpuState, r: CpuRegisters, m: Memory) = r { m.sh(RS_IMM16, RT) }
    inline fun sw(s: CpuState, r: CpuRegisters, m: Memory) = r { m.sw(RS_IMM16, RT) }
    inline fun sc(s: CpuState, r: CpuRegisters, m: Memory) = r { m.sw(RS_IMM16, RT); RT = 1 }

    inline fun lwc1(s: CpuState, r: CpuRegisters, m: Memory) = r { FT_I = m.lw(RS_IMM16) }
    inline fun swc1(s: CpuState, r: CpuRegisters, m: Memory) = r { m.sw(RS_IMM16, FT_I) }

    // Special
    inline fun syscall(s: CpuState, r: CpuRegisters, m: Memory) = r.preadvance { s.syscall(SYSCALL) }

    inline fun _break(s: CpuState, r: CpuRegisters, m: Memory) = r.preadvance { throw CpuBreakException(SYSCALL) }

    // Set less
    //fun slt(s: CpuState, r: CpuRegisters, m: CpuMemory) = s { RD = (RS < RT).toInt() }
    //fun sltu(s: CpuState, r: CpuRegisters, m: CpuMemory) = s { RD = (RS ult RT).toInt() }

    //fun slti(s: CpuState, r: CpuRegisters, m: CpuMemory) = s { RT = (RS < S_IMM16).toInt() }
    //fun sltiu(s: CpuState, r: CpuRegisters, m: CpuMemory) = s { RT = (RS ult S_IMM16).toInt() }

    inline fun slt(s: CpuState, r: CpuRegisters, m: Memory) = r { RD = s.slt(RS, RT) }
    inline fun sltu(s: CpuState, r: CpuRegisters, m: Memory) = r { RD = s.sltu(RS, RT) }
    inline fun slti(s: CpuState, r: CpuRegisters, m: Memory) = r { RT = s.slt(RS, S_IMM16) }
    inline fun sltiu(s: CpuState, r: CpuRegisters, m: Memory) = r { RT = s.sltu(RS, S_IMM16) }


    // Branch
    inline fun beq(s: CpuState, r: CpuRegisters, m: Memory) = r.branch { RS == RT }

    inline fun bne(s: CpuState, r: CpuRegisters, m: Memory) = r.branch { RS != RT }
    inline fun bltz(s: CpuState, r: CpuRegisters, m: Memory) = r.branch { RS < 0 }
    inline fun blez(s: CpuState, r: CpuRegisters, m: Memory) = r.branch { RS <= 0 }
    inline fun bgtz(s: CpuState, r: CpuRegisters, m: Memory) = r.branch { RS > 0 }
    inline fun bgez(s: CpuState, r: CpuRegisters, m: Memory) = r.branch { RS >= 0 }
    inline fun bgezal(s: CpuState, r: CpuRegisters, m: Memory) = r.branch { RA = nPC + 4; RS >= 0 }
    inline fun bltzal(s: CpuState, r: CpuRegisters, m: Memory) = r.branch { RA = nPC + 4; RS < 0 }

    inline fun beql(s: CpuState, r: CpuRegisters, m: Memory) = r.branchLikely { RS == RT }
    inline fun bnel(s: CpuState, r: CpuRegisters, m: Memory) = r.branchLikely { RS != RT }
    inline fun bltzl(s: CpuState, r: CpuRegisters, m: Memory) = r.branchLikely { RS < 0 }
    inline fun blezl(s: CpuState, r: CpuRegisters, m: Memory) = r.branchLikely { RS <= 0 }
    inline fun bgtzl(s: CpuState, r: CpuRegisters, m: Memory) = r.branchLikely { RS > 0 }
    inline fun bgezl(s: CpuState, r: CpuRegisters, m: Memory) = r.branchLikely { RS >= 0 }
    inline fun bgezall(s: CpuState, r: CpuRegisters, m: Memory) = r.branchLikely { RA = nPC + 4; RS >= 0 }
    inline fun bltzall(s: CpuState, r: CpuRegisters, m: Memory) = r.branchLikely { RA = nPC + 4; RS < 0 }

    inline fun bc1f(s: CpuState, r: CpuRegisters, m: Memory) = r.branch { !fcr31_cc }
    inline fun bc1t(s: CpuState, r: CpuRegisters, m: Memory) = r.branch { fcr31_cc }
    inline fun bc1fl(s: CpuState, r: CpuRegisters, m: Memory) = r.branchLikely { !fcr31_cc }
    inline fun bc1tl(s: CpuState, r: CpuRegisters, m: Memory) = r.branchLikely { fcr31_cc }

    //fun j(s: CpuState, r: CpuRegisters, m: CpuMemory) = r.none { _PC = _nPC; _nPC = (_PC and 0xf0000000.toInt()) or (JUMP_ADDRESS) } // @TODO: Kotlin.JS doesn't optimize 0xf0000000.toInt() and generates a long
    inline fun j(s: CpuState, r: CpuRegisters, m: Memory) = r.none { PC = nPC; nPC = (PC and (-268435456)) or (JUMP_ADDRESS) }

    inline fun jr(s: CpuState, r: CpuRegisters, m: Memory) = r.none { PC = nPC; nPC = RS }

    inline fun jal(s: CpuState, r: CpuRegisters, m: Memory) = r.none {
        j(s, r, m)
        RA = PC + 4
    } // $31 = PC + 8 (or nPC + 4); PC = nPC; nPC = (PC & 0xf0000000) | (target << 2);

    inline fun jalr(s: CpuState, r: CpuRegisters, m: Memory) = r.none { jr(s, r, m); RD = PC + 4; }

    // Float
    inline fun mfc1(s: CpuState, r: CpuRegisters, m: Memory) = r { RT = FS_I }

    inline fun mtc1(s: CpuState, r: CpuRegisters, m: Memory) = r { FS_I = RT }
    inline fun cvt_s_w(s: CpuState, r: CpuRegisters, m: Memory) = r { FD = FS_I.toFloat() }
    inline fun cvt_w_s(s: CpuState, r: CpuRegisters, m: Memory) = r {

        FD_I = when (this.fcr31_rm) {
            0 -> Math.rint(FS) // rint: round nearest
            1 -> Math.cast(FS) // round to zero
            2 -> Math.ceil(FS) // round up (ceil)
            3 -> Math.floor(FS) // round down (floor)
            else -> FS.toInt()
        }
    }

    inline fun trunc_w_s(s: CpuState, r: CpuRegisters, m: Memory) = r { FD_I = Math.trunc(FS) }
    inline fun round_w_s(s: CpuState, r: CpuRegisters, m: Memory) = r { FD_I = Math.round(FS) }
    inline fun ceil_w_s(s: CpuState, r: CpuRegisters, m: Memory) = r { FD_I = Math.ceil(FS) }
    inline fun floor_w_s(s: CpuState, r: CpuRegisters, m: Memory) = r { FD_I = Math.floor(FS) }

    inline fun CpuRegisters.checkNan(callback: CpuRegisters.() -> Unit) = this.normal {
        callback()
        if (FD.isNaN()) fcr31 = fcr31 or 0x00010040
        if (FD.isInfinite()) fcr31 = fcr31 or 0x00005014
    }

    inline fun mov_s(s: CpuState, r: CpuRegisters, m: Memory) = r.checkNan { FD = FS }
    inline fun add_s(s: CpuState, r: CpuRegisters, m: Memory) = r.checkNan { FD = FS pspAdd FT }
    inline fun sub_s(s: CpuState, r: CpuRegisters, m: Memory) = r.checkNan { FD = FS pspSub FT }
    inline fun mul_s(s: CpuState, r: CpuRegisters, m: Memory) = r.checkNan { FD = FS * FT; if (fcr31_fs && FD.isAlmostZero()) FD = 0f }
    inline fun div_s(s: CpuState, r: CpuRegisters, m: Memory) = r.checkNan { FD = FS / FT }
    inline fun neg_s(s: CpuState, r: CpuRegisters, m: Memory) = r.checkNan { FD = -FS }
    inline fun abs_s(s: CpuState, r: CpuRegisters, m: Memory) = r.checkNan { FD = kotlin.math.abs(FS) }
    inline fun sqrt_s(s: CpuState, r: CpuRegisters, m: Memory) = r.checkNan { FD = kotlin.math.sqrt(FS) }

    inline fun CpuRegisters._cu(callback: CpuRegisters.() -> Boolean) =
        this { fcr31_cc = if (FS.isNaN() || FT.isNaN()) true else callback() }

    inline fun CpuRegisters._co(callback: CpuRegisters.() -> Boolean) =
        this { fcr31_cc = if (FS.isNaN() || FT.isNaN()) false else callback() }

    inline fun c_f_s(s: CpuState, r: CpuRegisters, m: Memory) = r._co { false }
    inline fun c_un_s(s: CpuState, r: CpuRegisters, m: Memory) = r._cu { false }
    inline fun c_eq_s(s: CpuState, r: CpuRegisters, m: Memory) = r._co { FS == FT }
    inline fun c_ueq_s(s: CpuState, r: CpuRegisters, m: Memory) = r._cu { FS == FT }
    inline fun c_olt_s(s: CpuState, r: CpuRegisters, m: Memory) = r._co { FS < FT }
    inline fun c_ult_s(s: CpuState, r: CpuRegisters, m: Memory) = r._cu { FS < FT }
    inline fun c_ole_s(s: CpuState, r: CpuRegisters, m: Memory) = r._co { FS <= FT }
    inline fun c_ule_s(s: CpuState, r: CpuRegisters, m: Memory) = r._cu { FS <= FT }

    inline fun c_sf_s(s: CpuState, r: CpuRegisters, m: Memory) = r._co { false }
    inline fun c_ngle_s(s: CpuState, r: CpuRegisters, m: Memory) = r._cu { false }
    inline fun c_seq_s(s: CpuState, r: CpuRegisters, m: Memory) = r._co { FS == FT }
    inline fun c_ngl_s(s: CpuState, r: CpuRegisters, m: Memory) = r._cu { FS == FT }
    inline fun c_lt_s(s: CpuState, r: CpuRegisters, m: Memory) = r._co { FS < FT }
    inline fun c_nge_s(s: CpuState, r: CpuRegisters, m: Memory) = r._cu { FS < FT }
    inline fun c_le_s(s: CpuState, r: CpuRegisters, m: Memory) = r._co { FS <= FT }
    inline fun c_ngt_s(s: CpuState, r: CpuRegisters, m: Memory) = r._cu { FS <= FT }

    inline fun cfc1(s: CpuState, r: CpuRegisters, m: Memory) = r {
        when (IR.rd) {
            0 -> RT = fcr0
            25 -> RT = fcr25
            26 -> RT = fcr26
            27 -> RT = fcr27
            28 -> RT = fcr28
            31 -> RT = fcr31
            else -> RT = -1
        }
    }

    inline fun ctc1(s: CpuState, r: CpuRegisters, m: Memory) = r {
        when (IR.rd) {
            31 -> updateFCR31(RT)
        }
    }

    inline fun _lv_x(s: CpuState, r: CpuRegisters, m: Memory, size: Int) = r {
        getVectorRegisters(VSRC, IR.vt5_1, size)
        val start = RS_IMM14
        for (n in 0 until size) s.VFPRI[VSRC[n]] = m.lw(start + n * 4)
    }

    inline fun _sv_x(s: CpuState, r: CpuRegisters, m: Memory, size: Int) = s {
        getVectorRegisters(VSRC, IR.vt5_1, size)
        val start = RS_IMM14
        for (n in 0 until size) mem.sw(start + n * 4, s.VFPRI[VSRC[n]])
    }

    inline fun lv_s(s: CpuState, r: CpuRegisters, m: Memory) = _lv_x(s, r, m, 1)
    inline fun lv_q(s: CpuState, r: CpuRegisters, m: Memory) = _lv_x(s, r, m, 4)

    inline fun sv_s(s: CpuState, r: CpuRegisters, m: Memory) = _sv_x(s, r, m, 1)
    inline fun sv_q(s: CpuState, r: CpuRegisters, m: Memory) = _sv_x(s, r, m, 4)

    inline fun lvl_q(s: CpuState, r: CpuRegisters, m: Memory) = s {
        getVectorRegisters(VSRC, IR.vt5_1, 4)
        mem.lvl_q(RS_IMM14) { i, value -> s.setVfprI(VSRC[i], value) }
    }

    inline fun lvr_q(s: CpuState, r: CpuRegisters, m: Memory) = s {
        getVectorRegisters(VSRC, IR.vt5_1, 4)
        mem.lvr_q(RS_IMM14) { i, value -> s.setVfprI(VSRC[i], value) }
    }

    inline fun svl_q(s: CpuState, r: CpuRegisters, m: Memory) = s {
        getVectorRegisters(VSRC, IR.vt5_1, 4)
        mem.svl_q(RS_IMM14) { getVfprI(VSRC[it]) }
    }

    inline fun svr_q(s: CpuState, r: CpuRegisters, m: Memory) = s {
        getVectorRegisters(VSRC, IR.vt5_1, 4)
        mem.svr_q(RS_IMM14) { getVfprI(VSRC[it]) }
    }

    inline fun cc_8888_to_4444(i: Int): Int = 0 or
            (((i ushr 4) and 15) shl 0) or
            (((i ushr 12) and 15) shl 4) or
            (((i ushr 20) and 15) shl 8) or
            (((i ushr 28) and 15) shl 12)

    inline fun cc_8888_to_5551(i: Int): Int = 0 or
            (((i ushr 3) and 31) shl 0) or
            (((i ushr 11) and 31) shl 5) or
            (((i ushr 19) and 31) shl 10) or
            (((i ushr 31) and 1) shl 15)

    inline fun cc_8888_to_5650(i: Int): Int = 0 or
            (((i ushr 3) and 31) shl 0) or
            (((i ushr 10) and 63) shl 5) or
            (((i ushr 19) and 31) shl 11)

    inline fun CpuState._vtXXXX_q(func: (Int) -> Int) = this {
        setVDI_VS(destSize = IR.one_two / 2) {
            func(vsi[it * 2 + 0]) or (func(vsi[it * 2 + 1]) shl 16)
        }
    }

    inline fun vt4444_q(s: CpuState, r: CpuRegisters, m: Memory) = s._vtXXXX_q(this::cc_8888_to_4444)
    inline fun vt5551_q(s: CpuState, r: CpuRegisters, m: Memory) = s._vtXXXX_q(this::cc_8888_to_5551)
    inline fun vt5650_q(s: CpuState, r: CpuRegisters, m: Memory) = s._vtXXXX_q(this::cc_8888_to_5650)

    inline fun _vc2i(s: CpuState, func: (index: Int, value: Int) -> Int) = s {
        setVDI_VS(destSize = 4, srcSize = 1) { func(it, vsi.x) }
    }

    inline fun vc2i(s: CpuState, r: CpuRegisters, m: Memory) =
        _vc2i(s) { index, value -> (value shl ((3 - index) * 8)) and 0xFF000000.toInt() }

    inline fun vuc2i(s: CpuState, r: CpuRegisters, m: Memory) =
        _vc2i(s) { index, value -> ((((value ushr (index * 8)) and 0xFF) * 0x01010101) shr 1) and 0x80000000.toInt().inv() }

    inline fun _vs2i(s: CpuState, func: (index: Int, value: Int) -> Int) = s {
        setVDI_VS(destSize = IR.one_two * 2) { func(it % 2, vsi[it / 2]) }
    }

    inline fun vs2i(s: CpuState, r: CpuRegisters, m: Memory) = _vs2i(s) { index, value -> value.extract(index * 16, 16) shl 16 }
    inline fun vus2i(s: CpuState, r: CpuRegisters, m: Memory) = _vs2i(s) { index, value -> value.extract(index * 16, 16) shl 15 }

    inline fun _vi2c(s: CpuState, gen: (value: Int) -> Int) = s {
        setVDI_VS(destSize = 1, srcSize = 4) {
            RGBA.packFast(gen(vsi[0]), gen(vsi[1]), gen(vsi[2]), gen(vsi[3]))
        }
    }

    inline fun vi2c(s: CpuState, r: CpuRegisters, m: Memory) = _vi2c(s) { it.extract8(24) }
    inline fun vi2uc(s: CpuState, r: CpuRegisters, m: Memory) = _vi2c(s) { if (it < 0) 0 else it.extract8(23) }

    inline fun _vi2s(s: CpuState, gen: (value: Int) -> Int) = s {
        setVDI_VS(destSize = IR.one_two / 2) {
            val l = gen(vsi[it * 2 + 0])
            val r = gen(vsi[it * 2 + 1])
            l or (r shl 16)
        }
    }

    inline fun vi2s(s: CpuState, r: CpuRegisters, m: Memory) = _vi2s(s) { it ushr 16 }
    inline fun vi2us(s: CpuState, r: CpuRegisters, m: Memory) = _vi2s(s) { if (it < 0) 0 else it shr 15 }
    inline fun vi2f(s: CpuState, r: CpuRegisters, m: Memory) = s { setVD_VS { vsi[it] * 2f.pow(-IR.imm5) } }

    inline fun _vf2ix(s: CpuState, func: (value: Float, imm5: Int) -> Int) = s {
        setVDI_VS { if (vs[it].isNaN()) 0x7FFFFFFF else func(vs[it], IR.imm5) }
    }

    inline fun vf2id(s: CpuState, r: CpuRegisters, m: Memory) = _vf2ix(s) { value, imm5 -> floor(value * 2f.pow(imm5)).toInt() }
    inline fun vf2iu(s: CpuState, r: CpuRegisters, m: Memory) = _vf2ix(s) { value, imm5 -> ceil(value * 2f.pow(imm5)).toInt() }
    inline fun vf2in(s: CpuState, r: CpuRegisters, m: Memory) = _vf2ix(s) { value, imm5 -> Math.rint((value * 2f.pow(imm5))) }
    inline fun vf2iz(s: CpuState, r: CpuRegisters, m: Memory) = _vf2ix(s) { value, imm5 ->
        val rs = value * 2f.pow(imm5); if (value >= 0) floor(rs).toInt() else ceil(rs).toInt()
    }

    inline fun vf2h(s: CpuState, r: CpuRegisters, m: Memory) = s {
        setVDI_VS(destSize = IR.one_two / 2) {
            val l = HalfFloat.floatBitsToHalfFloatBits(vsi[it * 2 + 0])
            val r = HalfFloat.floatBitsToHalfFloatBits(vsi[it * 2 + 1])
            (l) or (r shl 16)
        }
    }

    inline fun vh2f(s: CpuState, r: CpuRegisters, m: Memory) = s {
        setVDI_VS(destSize = IR.one_two * 2) {
            HalfFloat.halfFloatBitsToFloatBits(vsi[it / 2].extract((it % 2) * 16, 16))
        }
    }

    inline fun viim(s: CpuState, r: CpuRegisters, m: Memory) = s { VT = S_IMM16.toFloat() }
    inline fun vfim(s: CpuState, r: CpuRegisters, m: Memory) = s { VT_I = HalfFloat.halfFloatBitsToFloatBits(U_IMM16) }

    inline fun vcst(s: CpuState, r: CpuRegisters, m: Memory) = s { VD = VfpuConstants[IR.imm5].value }
    inline fun mtv(s: CpuState, r: CpuRegisters, m: Memory) = s { VD_I = RT }
    inline fun vpfxt(s: CpuState, r: CpuRegisters, m: Memory) = s { vpfxt.setEnable(IR) }
    inline fun vpfxd(s: CpuState, r: CpuRegisters, m: Memory) = s { vpfxd.setEnable(IR) }
    inline fun vpfxs(s: CpuState, r: CpuRegisters, m: Memory) = s { vpfxs.setEnable(IR) }
    inline fun vavg(s: CpuState, r: CpuRegisters, m: Memory) = s {
        setVD_VS(destSize = 1, prefixes = true) {
            ((0 until vsSize).sumByFloat { (vs[it] / vsSize) })
        }
    }

    inline fun vfad(s: CpuState, r: CpuRegisters, m: Memory) = s {
        setVD_VS(destSize = 1, prefixes = true) {
            ((0 until vsSize).sumByFloat { vs[it] })
        }
    }

    inline fun vrot(s: CpuState, r: CpuRegisters, m: Memory) = s {
        val vectorSize = IR.one_two
        val imm5 = IR.imm5
        val cosIndex = imm5.extract(0, 2)
        val sinIndex = imm5.extract(2, 2)
        val negateSin = imm5.extractBool(4)

        setVD_VS(vectorSize, 1, prefixes = true) {
            var sine = sinv1(vs.x)
            val cosine = cosv1(vs.x)
            if (negateSin) sine = -sine

            when (it) {
                cosIndex -> cosine
                sinIndex -> sine
                else -> if (sinIndex == cosIndex) sine else 0f
            }
        }
    }

    // Vector operations (zero operands)
    inline fun vzero(s: CpuState, r: CpuRegisters, m: Memory) = s { setVD_(prefixes = true) { 0f } }
    inline fun vone(s: CpuState, r: CpuRegisters, m: Memory) = s { setVD_(prefixes = true) { 1f } }

    // Vector operations (one operand)
    inline fun vmov(s: CpuState, r: CpuRegisters, m: Memory) = s { setVD_VS(prefixes = true) { vs[it] } }
    inline fun vabs(s: CpuState, r: CpuRegisters, m: Memory) = s { setVD_VS(prefixes = true) { abs(vs[it]) } }
    inline fun vsqrt(s: CpuState, r: CpuRegisters, m: Memory) = s { setVD_VS(prefixes = true) { sqrt(vs[it]) } }
    inline fun vneg(s: CpuState, r: CpuRegisters, m: Memory) = s { setVD_VS(prefixes = true) { -vs[it] } }
    inline fun vsat0(s: CpuState, r: CpuRegisters, m: Memory) = s { setVD_VS(prefixes = true) { vs[it].pspSat0 } }
    inline fun vsat1(s: CpuState, r: CpuRegisters, m: Memory) = s { setVD_VS(prefixes = true) { vs[it].pspSat1 } }
    inline fun vrcp(s: CpuState, r: CpuRegisters, m: Memory) = s { setVD_VS(prefixes = true) { 1f / vs[it] } }
    inline fun vrsq(s: CpuState, r: CpuRegisters, m: Memory) = s { setVD_VS(prefixes = true) { 1f / sqrt(vs[it]) } }
    inline fun vsin(s: CpuState, r: CpuRegisters, m: Memory) = s { setVD_VS(prefixes = true) { sinv1(vs[it]) } }
    inline fun vasin(s: CpuState, r: CpuRegisters, m: Memory) = s { setVD_VS(prefixes = true) { asinv1(vs[it]) } }
    inline fun vnsin(s: CpuState, r: CpuRegisters, m: Memory) = s { setVD_VS(prefixes = true) { -sinv1(vs[it]) } }
    inline fun vcos(s: CpuState, r: CpuRegisters, m: Memory) = s { setVD_VS(prefixes = true) { cosv1(vs[it]) } }
    inline fun vexp2(s: CpuState, r: CpuRegisters, m: Memory) = s { setVD_VS(prefixes = true) { 2f.pow(vs[it]) } }
    inline fun vrexp2(s: CpuState, r: CpuRegisters, m: Memory) = s { setVD_VS(prefixes = true) { 1f / 2f.pow(vs[it]) } }
    inline fun vlog2(s: CpuState, r: CpuRegisters, m: Memory) = s { setVD_VS(prefixes = true) { log2(vs[it]) } }
    inline fun vnrcp(s: CpuState, r: CpuRegisters, m: Memory) = s { setVD_VS(prefixes = true) { -1f / vs[it] } }
    inline fun vsgn(s: CpuState, r: CpuRegisters, m: Memory) = s { setVD_VS(prefixes = true) { vs[it].pspSign } }
    inline fun vocp(s: CpuState, r: CpuRegisters, m: Memory) = s { setVD_VS(prefixes = true) { 1f - vs[it] } }
    inline fun vbfy1(s: CpuState, r: CpuRegisters, m: Memory) = s {
        setVD_VS(prefixes = true) {
            when (it) {
                0 -> vs.x + vs.y
                1 -> vs.x - vs.y
                2 -> vs.z + vs.w
                3 -> vs.z - vs.w
                else -> invalidOp
            }
        }
    }

    inline fun vbfy2(s: CpuState, r: CpuRegisters, m: Memory) = s {
        setVD_VS(prefixes = true) {
            when (it) {
                0 -> vs.x + vs.z
                1 -> vs.y + vs.w
                2 -> vs.x - vs.z
                3 -> vs.y - vs.w
                else -> invalidOp
            }
        }
    }

    inline fun vsrt1(s: CpuState, r: CpuRegisters, m: Memory) = s {
        setVD_VS(prefixes = true) {
            when (it) {
                0 -> min(vs.x, vs.y)
                1 -> max(vs.x, vs.y)
                2 -> min(vs.z, vs.w)
                3 -> max(vs.z, vs.w)
                else -> invalidOp
            }
        }
    }

    inline fun vsrt2(s: CpuState, r: CpuRegisters, m: Memory) = s {
        setVD_VS(prefixes = true) {
            vs.run {
                when (it) {
                    0 -> min(x, w)
                    1 -> min(y, z)
                    2 -> max(y, z)
                    3 -> max(x, w)
                    else -> invalidOp
                }
            }
        }
    }

    inline fun vsrt3(s: CpuState, r: CpuRegisters, m: Memory) = s {
        setVD_VS(prefixes = true) {
            when (it) {
                0 -> max(vs.x, vs.y)
                1 -> min(vs.x, vs.y)
                2 -> max(vs.z, vs.w)
                3 -> min(vs.z, vs.w)
                else -> invalidOp
            }
        }
    }

    inline fun vsrt4(s: CpuState, r: CpuRegisters, m: Memory) = s {
        setVD_VS(prefixes = true) {
            when (it) {
                0 -> max(vs.x, vs.w)
                1 -> max(vs.y, vs.z)
                2 -> min(vs.y, vs.z)
                3 -> min(vs.x, vs.w)
                else -> invalidOp
            }
        }
    }

    // Vector operations (two operands)
    inline fun vsge(s: CpuState, r: CpuRegisters, m: Memory) = s { setVD_VSVT(prefixes = true) { if (vs[it] >= vt[it]) 1f else 0f } }
    inline fun vslt(s: CpuState, r: CpuRegisters, m: Memory) = s { setVD_VSVT(prefixes = true) { if (vs[it] < vt[it]) 1f else 0f } }
    inline fun vscmp(s: CpuState, r: CpuRegisters, m: Memory) = s { setVD_VSVT(prefixes = true) { vs[it].compareTo(vt[it]).toFloat() } }

    inline fun vadd(s: CpuState, r: CpuRegisters, m: Memory) = s { setVD_VSVT(prefixes = true) { vs[it] pspAdd vt[it] } }
    inline fun vsub(s: CpuState, r: CpuRegisters, m: Memory) = s { setVD_VSVT(prefixes = true) { vs[it] pspSub vt[it] } }
    inline fun vmul(s: CpuState, r: CpuRegisters, m: Memory) = s { setVD_VSVT(prefixes = true) { vs[it] * vt[it] } }
    inline fun vdiv(s: CpuState, r: CpuRegisters, m: Memory) = s { setVD_VSVT(prefixes = true) { vs[it] / vt[it] } }
    inline fun vmin(s: CpuState, r: CpuRegisters, m: Memory) = s { setVD_VSVT(prefixes = true) { min(vs[it], vt[it]) } }
    inline fun vmax(s: CpuState, r: CpuRegisters, m: Memory) = s { setVD_VSVT(prefixes = true) { max(vs[it], vt[it]) } }
    inline fun vcrs_t(s: CpuState, r: CpuRegisters, m: Memory) = s {
        setVD_VSVT(prefixes = true) {
            when (it) {
                0 -> vs.y * vt.z
                1 -> vs.z * vt.x
                2 -> vs.x * vt.y
                else -> invalidOp
            }
        }
    }

    inline fun vcrsp_t(s: CpuState, r: CpuRegisters, m: Memory) = s {
        setVD_VSVT(prefixes = true) {
            when (it) {
                0 -> +vs.y * vt.z - vs.z * vt.y
                1 -> +vs.z * vt.x - vs.x * vt.z
                2 -> +vs.x * vt.y - vs.y * vt.x
                else -> invalidOp
            }
        }
    }

    inline fun vqmul(s: CpuState, r: CpuRegisters, m: Memory) = s {
        setVD_VSVT(prefixes = true) {
            when (it) {
                0 -> +vs.x * vt.w + vs.y * vt.z - vs.z * vt.y + vs.w * vt.x
                1 -> -vs.x * vt.z + vs.y * vt.w + vs.z * vt.x + vs.w * vt.y
                2 -> +vs.x * vt.y - vs.y * vt.x + vs.z * vt.w + vs.w * vt.z
                3 -> -vs.x * vt.x - vs.y * vt.y - vs.z * vt.z + vs.w * vt.w
                else -> invalidOp
            }
        }
    }

    inline fun vdot(s: CpuState, r: CpuRegisters, m: Memory) = s {
        setVD_VSVT(destSize = 1, prefixes = true) {
            ((0 until vsSize).sumByFloat { (vs[it] * vt[it]) })
        }
    }

    inline fun vscl(s: CpuState, r: CpuRegisters, m: Memory) = s { setVD_VSVT(targetSize = 1, prefixes = true) { vs[it] * vt.x } }

    inline fun vhdp(s: CpuState, r: CpuRegisters, m: Memory) = s {
        setVD_VSVT(destSize = 1, prefixes = true) {
            vs[vsSize - 1] = 1f
            (0 until vsSize).sumByFloat { (vs[it] * vt[it]) }
        }
    }

    inline fun vdet(s: CpuState, r: CpuRegisters, m: Memory) = s {
        setVD_VSVT(destSize = 1, prefixes = true) {
            vs.x * vt.y - vs.y * vt.x
        }
    }

    inline fun vcmp(s: CpuState, r: CpuRegisters, m: Memory) = s {
        val size = IR.one_two
        var cc = 0
        _VSVT(prefixes = true) {
            val cond = when (IR.imm4) {
                VCondition.FL -> false
                VCondition.EQ -> vs[it] == vt[it]
                VCondition.LT -> vs[it] < vt[it]
                VCondition.LE -> vs[it] <= vt[it]

                VCondition.TR -> true
                VCondition.NE -> vs[it] != vt[it]
                VCondition.GE -> vs[it] >= vt[it]
                VCondition.GT -> vs[it] > vt[it]

                VCondition.EZ -> (vs[it] == 0f) || (vs[it] == -0f)
                VCondition.EN -> vs[it].isNaN()
                VCondition.EI -> vs[it].isInfinite()
                VCondition.ES -> vs[it].isNanOrInfinitef()

                VCondition.NZ -> vs[it] != 0f
                VCondition.NN -> !(vs[it].isNaN())
                VCondition.NI -> !(vs[it].isInfinite())
                VCondition.NS -> !(vs[it].isNanOrInfinitef())

                else -> false
            }

            if (cond) {
                cc = cc or (1 shl it)
            }
        }
        val mask = size.mask()
        val affectedBits = (mask or (1 shl 4) or (1 shl 5))
        if ((cc and mask) != 0) cc = cc.insert(true, 4)
        if ((cc and mask) == mask) cc = cc.insert(true, 5)

        VFPR_CC = (VFPR_CC and affectedBits.inv()) or cc
        //println("vcmp:$cc --> $VFPR_CC")
    }

    inline fun _vcmovtf(s: CpuState, truth: Boolean) = s {
        val ccidx = IR.imm3
        //println("_vcmovtf($truth)[$ccidx]")
        setVD_VDVS(prefixes = true) {
            val cond = when (ccidx) {
                0, 1, 2, 3, 4, 5 -> VFPR_CC(ccidx)
                6 -> VFPR_CC(it)
                7 -> true
                else -> false
            }
            if (cond != truth) vs[it] else vd[it]
        }
    }

    inline fun vcmovf(s: CpuState, r: CpuRegisters, m: Memory) = _vcmovtf(s, true)
    inline fun vcmovt(s: CpuState, r: CpuRegisters, m: Memory) = _vcmovtf(s, false)

    inline fun vwbn(s: CpuState, r: CpuRegisters, m: Memory) = s {
        setVDI_VS(destSize = 1) {
            val exp = IR.imm8
            val sigbit = vsi[it] and 0x80000000.toInt()
            val prevExp = (vsi[it] and 0x7F800000) ushr 23
            var mantissa = (vsi[it] and 0x007FFFFF) or 0x00800000
            if (prevExp != 0xFF && prevExp != 0) {
                if (exp > prevExp) {
                    val shift = (exp - prevExp) and 0xF
                    mantissa = mantissa ushr shift
                } else {
                    val shift = (prevExp - exp) and 0xF
                    mantissa = mantissa shl shift
                }
                sigbit or (mantissa and 0x007FFFFF) or (exp shl 23)
            } else {
                vsi[it] or (exp shl 23)
            }
        }
    }

    inline fun vsbn(s: CpuState, r: CpuRegisters, m: Memory) = s {
        setVDI_VSVT(targetSize = 1) {
            val exp = (127 + vti[0]) and 0xFF
            val prev = vsi[it] and 0x7F800000
            if (prev != 0 && prev != 0x7F800000) {
                (vsi[it] and 0x7F800000.inv()) or (exp shl 23)
            } else {
                vsi[it]
            }

            //scalab(vs[it], vti[it])
        }
    }

    // Matrix operations
    inline fun vmzero(s: CpuState, r: CpuRegisters, m: Memory) = s { setMatrixVD { 0f } }

    inline fun vmone(s: CpuState, r: CpuRegisters, m: Memory) = s { setMatrixVD { 1f } }
    inline fun vmidt(s: CpuState, r: CpuRegisters, m: Memory) = s { setMatrixVD { if (row == col) 1f else 0f } }
    inline fun vmmov(s: CpuState, r: CpuRegisters, m: Memory) = s { setMatrixVD_VS { ms[col, row] } }
    inline fun vmmul(s: CpuState, r: CpuRegisters, m: Memory) = s {
        setMatrixVD_VSVT {
            (0 until side).map { ms[row, it] * mt[col, it] }.sum()
        }
    }

    inline fun mfvc(s: CpuState, r: CpuRegisters, m: Memory) = s { RT = r.getVfprC(IR.imm7) }
    inline fun mtvc(s: CpuState, r: CpuRegisters, m: Memory) = s { r.setVfprC(IR.imm7, RT) }

    inline fun _vtfm_x(s: CpuState, size: Int) = s {
        vfpuContext.run {
            getVectorRegisterValues(b_vt, IR.vt, size)

            for (n in 0 until size) {
                getVectorRegisterValues(b_vs, IR.vs + n, size)
                vfpuContext.vd[n] = (0 until size).sumByFloat { (vs[it] * vt[it]) }
            }

            setVectorRegisterValues(b_vd, IR.vd, size)
        }
    }

    inline fun _vhtfm_x(s: CpuState, size: Int) = s {
        vfpuContext.run {
            getVectorRegisterValues(b_vt, IR.vt, size - 1)

            vt[size - 1] = 1f
            for (n in 0 until size) {
                getVectorRegisterValues(b_vs, IR.vs + n, size)
                vfpuContext.vd[n] = (0 until size).sumByFloat { (vs[it] * vt[it]) }
            }

            setVectorRegisterValues(b_vd, IR.vd, size)
        }
    }

    inline fun vtfm2(s: CpuState, r: CpuRegisters, m: Memory) = _vtfm_x(s, 2)
    inline fun vtfm3(s: CpuState, r: CpuRegisters, m: Memory) = _vtfm_x(s, 3)
    inline fun vtfm4(s: CpuState, r: CpuRegisters, m: Memory) = _vtfm_x(s, 4)

    inline fun vhtfm2(s: CpuState, r: CpuRegisters, m: Memory) = _vhtfm_x(s, 2)
    inline fun vhtfm3(s: CpuState, r: CpuRegisters, m: Memory) = _vhtfm_x(s, 3)
    inline fun vhtfm4(s: CpuState, r: CpuRegisters, m: Memory) = _vhtfm_x(s, 4)

    inline fun vmscl(s: CpuState, r: CpuRegisters, m: Memory) = s {
        val scale = vfpuContext.sreadVt(s, size = 1)[0]
        setMatrixVD_VS { ms[col, row] * scale }
    }

    inline fun vidt(s: CpuState, r: CpuRegisters, m: Memory) = unimplemented(s, Instructions.vidt)
    inline fun vnop(s: CpuState, r: CpuRegisters, m: Memory) = unimplemented(s, Instructions.vnop)
    inline fun vsync(s: CpuState, r: CpuRegisters, m: Memory) = unimplemented(s, Instructions.vsync)
    inline fun vflush(s: CpuState, r: CpuRegisters, m: Memory) = unimplemented(s, Instructions.vflush)
    inline fun vrnds(s: CpuState, r: CpuRegisters, m: Memory) = unimplemented(s, Instructions.vrnds)
    inline fun vrndi(s: CpuState, r: CpuRegisters, m: Memory) = unimplemented(s, Instructions.vrndi)
    inline fun vrndf1(s: CpuState, r: CpuRegisters, m: Memory) = unimplemented(s, Instructions.vrndf1)
    inline fun vrndf2(s: CpuState, r: CpuRegisters, m: Memory) = unimplemented(s, Instructions.vrndf2)
    inline fun vmfvc(s: CpuState, r: CpuRegisters, m: Memory) = unimplemented(s, Instructions.vmfvc)
    inline fun vmtvc(s: CpuState, r: CpuRegisters, m: Memory) = unimplemented(s, Instructions.vmtvc)
    inline fun mfvme(s: CpuState, r: CpuRegisters, m: Memory) = unimplemented(s, Instructions.mfvme)
    inline fun mtvme(s: CpuState, r: CpuRegisters, m: Memory) = unimplemented(s, Instructions.mtvme)
    inline fun vlgb(s: CpuState, r: CpuRegisters, m: Memory) = unimplemented(s, Instructions.vlgb)
    inline fun vsbz(s: CpuState, r: CpuRegisters, m: Memory) = unimplemented(s, Instructions.vsbz)
    inline fun vsocp(s: CpuState, r: CpuRegisters, m: Memory) = unimplemented(s, Instructions.vsocp)
    inline fun bvf(s: CpuState, r: CpuRegisters, m: Memory) = unimplemented(s, Instructions.bvf)
    inline fun bvt(s: CpuState, r: CpuRegisters, m: Memory) = unimplemented(s, Instructions.bvt)
    inline fun bvfl(s: CpuState, r: CpuRegisters, m: Memory) = unimplemented(s, Instructions.bvfl)
    inline fun bvtl(s: CpuState, r: CpuRegisters, m: Memory) = unimplemented(s, Instructions.bvtl)

    // Vectorial utilities

    inline fun CpuState.getMatrixRegsValues(out: FloatArray2, matrixReg: Int, N: Int): Int {
        val side = getMatrixRegs(tempRegs2, matrixReg, N)
        for (j in 0 until side) for (i in 0 until side) out[j, i] = getVfpr(tempRegs2[j, i])
        return side
    }

    inline fun CpuState.setMatrixRegsValues(inp: FloatArray2, matrixReg: Int, N: Int): Int {
        val side = getMatrixRegs(tempRegs2, matrixReg, N)
        for (j in 0 until side) for (i in 0 until side) setVfpr(tempRegs2[j, i], inp[j, i])
        return side
    }

    inline fun getMatrixRegs(out: IntArray2, matrixReg: Int, N: Int): Int {
        val side = N
        val mtx = (matrixReg ushr 2) and 7
        val col = matrixReg and 3
        val transpose = ((matrixReg ushr 5) and 1) != 0
        val row = when (N) {
            2 -> (matrixReg ushr 5) and 2
            3 -> (matrixReg ushr 6) and 1
            4 -> (matrixReg ushr 5) and 2
            else -> invalidOp
        }

        for (i in 0 until side) {
            for (j in 0 until side) {
                out[j, i] = (mtx * 4) + if (transpose) {
                    ((row + i) and 3) + ((col + j) and 3) * 32
                } else {
                    ((col + j) and 3) + ((row + i) and 3) * 32
                }
            }
        }

        return side
    }

    class MatrixContext {
        var side: Int = 0
        var col: Int = 0
        var row: Int = 0
        val ms = FloatArray2(4, 4)
        val md = FloatArray2(4, 4)
        val mt = FloatArray2(4, 4)
        fun setPos(c: Int, r: Int) = this.apply { col = c; row = r }
    }

    inline fun CpuState.setMatrixVD(side: Int = IR.one_two, callback: MatrixContext.() -> Float) {
        mc.side = side
        for (col in 0 until side) for (row in 0 until side) {
            mc.md[col, row] = callback(mc.setPos(col, row))
        }
        setMatrixRegsValues(mc.md, IR.vd, side)
    }

    inline fun CpuState.setMatrixVD_VS(side: Int = IR.one_two, callback: MatrixContext.() -> Float) {
        getMatrixRegsValues(mc.ms, IR.vs, side)

        mc.side = side
        for (col in 0 until side) for (row in 0 until side) {
            mc.md[col, row] = callback(mc.setPos(col, row))
        }
        setMatrixRegsValues(mc.md, IR.vd, side)
    }

    inline fun CpuState.setMatrixVD_VSVT(side: Int = IR.one_two, callback: MatrixContext.() -> Float) {
        getMatrixRegsValues(mc.ms, IR.vs, side)
        getMatrixRegsValues(mc.mt, IR.vt, side)

        mc.side = side
        for (col in 0 until side) for (row in 0 until side) {
            mc.md[col, row] = callback(mc.setPos(col, row))
        }
        setMatrixRegsValues(mc.md, IR.vd, side)
    }

    inline fun getVectorRegisters(out: IntArray, vectorReg: Int, N: Int): Int {
        val mtx = vectorReg.extract(2, 3)
        val col = vectorReg.extract(0, 2)
        val row: Int
        val length: Int = N
        val transpose = (N != 1) && vectorReg.extractBool(5)

        when (N) {
            1 -> row = (vectorReg ushr 5) and 3
            2 -> row = (vectorReg ushr 5) and 2
            3 -> row = (vectorReg ushr 6) and 1
            4 -> row = (vectorReg ushr 5) and 2
            else -> invalidOp
        }

        for (i in 0 until length) {
            out[i] = mtx * 4 + if (transpose) {
                ((row + i) and 3) + col * 32
            } else {
                col + ((row + i) and 3) * 32
            }
        }
        return N
    }

    inline fun CpuState.getVectorRegisterValues(out: FloatIntBuffer, vectorReg: Int, N: Int) {
        getVectorRegisters(tempRegs, vectorReg, N)
        for (n in 0 until N) out.i[n] = getVfprI(tempRegs[n])
    }

    inline fun CpuState.setVectorRegisterValues(inp: FloatIntBuffer, vectorReg: Int, N: Int) {
        getVectorRegisters(tempRegs, vectorReg, N)
        for (n in 0 until N) setVfprI(tempRegs[n], inp.i[n])
    }

    inner class VfpuContext {
        val b_vs = FloatIntBuffer(16)
        val b_vd = FloatIntBuffer(16)
        val b_vt = FloatIntBuffer(16)

        var vsSize: Int = 0
        var vdSize: Int = 0
        var vtSize: Int = 0

        val vs = b_vs.f
        val vd = b_vd.f
        val vt = b_vt.f

        val vsi = b_vs.i
        val vdi = b_vd.i
        val vti = b_vt.i

        var Float32Buffer.x: Float get() = this[0]; set(value) = run { this[0] = value }
        var Float32Buffer.y: Float get() = this[1]; set(value) = run { this[1] = value }
        var Float32Buffer.z: Float get() = this[2]; set(value) = run { this[2] = value }
        var Float32Buffer.w: Float get() = this[3]; set(value) = run { this[3] = value }

        var Int32Buffer.x: Int get() = this[0]; set(value) = run { this[0] = value }
        var Int32Buffer.y: Int get() = this[1]; set(value) = run { this[1] = value }
        var Int32Buffer.z: Int get() = this[2]; set(value) = run { this[2] = value }
        var Int32Buffer.w: Int get() = this[3]; set(value) = run { this[3] = value }

        fun CpuState.readVs(reg: Int = IR.vs, size: Int = IR.one_two, prefixes: Boolean = false): Float32Buffer {
            vsSize = size
            getVectorRegisterValues(b_vs, reg, size)
            if (prefixes) this.vpfxs.applyAndConsume(vs, size = size)
            return vs
        }

        fun CpuState.readVt(reg: Int = IR.vt, size: Int = IR.one_two, prefixes: Boolean = false): Float32Buffer {
            vtSize = size
            getVectorRegisterValues(b_vt, reg, size)
            if (prefixes) this.vpfxt.applyAndConsume(vt, size = size)
            return vt
        }

        fun CpuState.readVd(reg: Int = IR.vd, size: Int = IR.one_two): Float32Buffer {
            vdSize = size
            getVectorRegisterValues(b_vd, reg, size)
            return vd
        }

        fun CpuState.writeVd(reg: Int = IR.vd, size: Int = IR.one_two, prefixes: Boolean = false) {
            getVectorRegisters(tempRegs, reg, size)
            if (prefixes) {
                for (i in 0 until size) {
                    if (vpfxd.mustWrite(i)) setVfpr(tempRegs[i], vpfxd.transform(i, vd[i]))
                }
                vpfxd.consume()
            } else {
                for (i in 0 until size) {
                    setVfprI(tempRegs[i], vdi[i])
                }
            }
        }

        fun sreadVs(s: CpuState, reg: Int = s.IR.vs, size: Int = s.IR.one_two) = s.readVs(reg, size)
        fun sreadVt(s: CpuState, reg: Int = s.IR.vt, size: Int = s.IR.one_two) = s.readVt(reg, size)
        fun sreadVd(s: CpuState, reg: Int = s.IR.vd, size: Int = s.IR.one_two) = s.readVd(reg, size)
        fun swriteVd(s: CpuState, reg: Int = s.IR.vd, size: Int = s.IR.one_two) = s.writeVd(reg, size)
    }

    val vfpuContext = VfpuContext()

    inline fun CpuState.setVD_(
        destSize: Int = IR.one_two,
        prefixes: Boolean = false,
        callback: VfpuContext.(i: Int) -> Float
    ) = vfpuContext.run {
        vdSize = destSize
        for (n in 0 until destSize) vd[n] = callback(vfpuContext, n)
        writeVd(size = destSize, prefixes = prefixes)
        consumePrefixes(prefixes)
    }

    inline fun CpuState.setVDI_(
        destSize: Int = IR.one_two,
        prefixes: Boolean = false,
        callback: VfpuContext.(i: Int) -> Int
    ) = vfpuContext.run {
        vdSize = destSize
        for (n in 0 until destSize) vdi[n] = callback(vfpuContext, n)
        writeVd(size = destSize, prefixes = prefixes)
        consumePrefixes(prefixes)
    }

    //inline // @TODO: kotlin-native bug: https://github.com/JetBrains/kotlin-native/issues/1777
    fun CpuState.setVD_VS(
        destSize: Int = IR.one_two,
        srcSize: Int = IR.one_two,
        prefixes: Boolean = false,
        callback: VfpuContext.(i: Int) -> Float
    ) = vfpuContext.run {
        vdSize = destSize
        readVs(size = srcSize, prefixes = prefixes)
        for (n in 0 until destSize) vd[n] = callback(vfpuContext, n)
        writeVd(size = destSize, prefixes = prefixes)
        consumePrefixes(prefixes)
    }

    inline fun CpuState.setVDI_VS(
        destSize: Int = IR.one_two,
        srcSize: Int = IR.one_two,
        prefixes: Boolean = false,
        callback: VfpuContext.(i: Int) -> Int
    ) = vfpuContext.run {
        vdSize = destSize
        readVs(size = srcSize, prefixes = prefixes)
        for (n in 0 until destSize) vdi[n] = callback(vfpuContext, n)
        writeVd(size = destSize, prefixes = prefixes)
        consumePrefixes(prefixes)
    }

    inline fun CpuState.setVD_VDVS(
        destSize: Int = IR.one_two,
        srcSize: Int = IR.one_two,
        prefixes: Boolean = false,
        callback: VfpuContext.(i: Int) -> Float
    ) = vfpuContext.run {
        vdSize = destSize
        readVs(size = srcSize, prefixes = prefixes)
        readVd(size = destSize)
        for (n in 0 until destSize) vd[n] = callback(vfpuContext, n)
        writeVd(size = destSize, prefixes = prefixes)
        consumePrefixes(prefixes)
    }

    inline fun CpuState.setVD_VSVT(
        destSize: Int = IR.one_two,
        srcSize: Int = IR.one_two,
        targetSize: Int = srcSize,
        prefixes: Boolean = false,
        callback: VfpuContext.(i: Int) -> Float
    ) {
        vfpuContext.run {
            vfpuContext.vdSize = destSize
            readVs(size = srcSize, prefixes = prefixes)
            readVt(size = targetSize, prefixes = prefixes)
            for (n in 0 until destSize) vd[n] = callback(vfpuContext, n)
            writeVd(size = destSize, prefixes = prefixes)
            consumePrefixes(prefixes)
        }
    }

    inline fun CpuState.setVDI_VSVT(
        destSize: Int = IR.one_two,
        srcSize: Int = IR.one_two,
        targetSize: Int = srcSize,
        prefixes: Boolean = false,
        callback: VfpuContext.(i: Int) -> Int
    ) = vfpuContext.run {
        vfpuContext.vdSize = destSize
        readVs(size = srcSize, prefixes = prefixes)
        readVt(size = targetSize, prefixes = prefixes)
        for (n in 0 until destSize) vdi[n] = callback(vfpuContext, n)
        writeVd(size = destSize, prefixes = prefixes)
        consumePrefixes(prefixes)
    }

    inline fun CpuState._VSVT(
        destSize: Int = IR.one_two,
        srcSize: Int = IR.one_two,
        targetSize: Int = srcSize,
        prefixes: Boolean = false,
        callback: VfpuContext.(i: Int) -> Unit
    ) = vfpuContext.run {
        readVs(size = srcSize, prefixes = prefixes)
        readVt(size = targetSize, prefixes = prefixes)
        for (n in 0 until destSize) callback(vfpuContext, n)
        consumePrefixes(prefixes)
    }

    @PublishedApi
    internal fun CpuState.consumePrefixes(prefixes: Boolean) {
        if (prefixes) {
            vpfxs.consume()
            vpfxt.consume()
            vpfxd.consume()
        }
    }

    enum class VfpuConstants(val value: Float) {
        VFPU_ZERO(0f),
        VFPU_HUGE(340282346638528859811704183484516925440f),
        VFPU_SQRT2(sqrt(2f)),
        VFPU_SQRT1_2(sqrt(1f / 2f)),
        VFPU_2_SQRTPI(2f / sqrt(PI)),
        VFPU_2_PI((2f / PI).toFloat()),
        VFPU_1_PI((1f / PI).toFloat()),
        VFPU_PI_4(PI / 4f),
        VFPU_PI_2(PI / 2f),
        VFPU_PI(PI),
        VFPU_E(E),
        VFPU_LOG2E(log2(E)),
        VFPU_LOG10E(log10(E)),
        VFPU_LN2(log(2.0, E)),
        VFPU_LN10(log(10.0, E)),
        VFPU_2PI(2f * PI),
        VFPU_PI_6(PI / 6.0),
        VFPU_LOG10TWO(log10(2f)),
        VFPU_LOG2TEN(log2(10f)),
        VFPU_SQRT3_2(sqrt(3f) / 2f);

        // @TODO: Kotlin Bug. It is used!
        @Suppress("unused")
        constructor(value: Double) : this(value.toFloat())

        companion object {
            val values = values()
            operator fun get(index: Int) = values[index]
        }
    }

    // Not implemented instructions!

    fun cache(s: CpuState, r: CpuRegisters, m: Memory) = unimplemented(s, Instructions.cache)
    fun sync(s: CpuState, r: CpuRegisters, m: Memory) = unimplemented(s, Instructions.sync)
    fun dbreak(s: CpuState, r: CpuRegisters, m: Memory) = unimplemented(s, Instructions.dbreak)
    fun halt(s: CpuState, r: CpuRegisters, m: Memory) = unimplemented(s, Instructions.halt)
    fun dret(s: CpuState, r: CpuRegisters, m: Memory) = unimplemented(s, Instructions.dret)
    fun eret(s: CpuState, r: CpuRegisters, m: Memory) = unimplemented(s, Instructions.eret)
    fun mfdr(s: CpuState, r: CpuRegisters, m: Memory) = unimplemented(s, Instructions.mfdr)
    fun mtdr(s: CpuState, r: CpuRegisters, m: Memory) = unimplemented(s, Instructions.mtdr)
    fun cfc0(s: CpuState, r: CpuRegisters, m: Memory) = unimplemented(s, Instructions.cfc0)
    fun ctc0(s: CpuState, r: CpuRegisters, m: Memory) = unimplemented(s, Instructions.ctc0)
    fun mfc0(s: CpuState, r: CpuRegisters, m: Memory) = unimplemented(s, Instructions.mfc0)
    fun mtc0(s: CpuState, r: CpuRegisters, m: Memory) = unimplemented(s, Instructions.mtc0)
    fun mfv(s: CpuState, r: CpuRegisters, m: Memory) = unimplemented(s, Instructions.mfv)
}

