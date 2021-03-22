package com.soywiz.kpspemu.cpu.interpreter

import com.soywiz.klogger.*
import com.soywiz.kmem.*
import com.soywiz.korim.color.*
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
import com.soywiz.korma.math.isAlmostZero
import com.soywiz.krypto.encoding.*
import com.soywiz.korio.error.invalidOp as invalidOp1

class CpuInterpreter(
    var cpu: CpuState,
    val breakpoints: Breakpoints,
    val nameProvider: NameProvider,
    var trace: Boolean = false
) {
    val dispatcher = InstructionDispatcher(InstructionInterpreter(cpu))

    fun steps(count: Int, trace: Boolean = false): Int {
        val mem = cpu.mem.getFastMem()
        //val mem = null
        return if (mem != null) {
            stepsFastMem(mem, cpu.mem.getFastMemOffset(MemoryInfo.MAIN_OFFSET) - MemoryInfo.MAIN_OFFSET, count, trace)
        } else {
            stepsNormal(count, trace)
        }
    }

    fun stepsNormal(count: Int, trace: Boolean): Int {
        val dispatcher = this.dispatcher
        val cpu = this.cpu
        val mem = cpu.mem
        var sPC = 0
        var n = 0
        //val fast = (mem as FastMemory).buffer
        val breakpointsEnabled = breakpoints.enabled
        try {
            while (n < count) {
                sPC = cpu._PC
                checkTrace(sPC, cpu)
                n++
                //if (PC == 0) throw IllegalStateException("Trying to execute PC=0")
                if (trace) tracePC()
                val IR = mem.lw(sPC)
                //val IR = fast.getAlignedInt32((PC ushr 2) and Memory.MASK)
                cpu.IR = IR
                dispatcher.dispatch(cpu, sPC, IR)
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
        //if (sPC in 0x08900DFC..0x08900E38) {
        //    doTrace(sPC, cpu)
        //}
        //doTrace(sPC, cpu)
    }

    fun stepsFastMem(mem: KmlNativeBuffer, memOffset: Int, count: Int, trace: Boolean): Int {
        val i32 = mem.i32
        val cpu = this.cpu
        var n = 0
        var sPC = 0
        try {
            while (n < count) {
                sPC = cpu._PC and 0x0FFFFFFF
                checkTrace(sPC, cpu)
                n++
                val IR = i32[(memOffset + sPC) ushr 2]
                cpu.IR = IR
                dispatcher.dispatch(cpu, sPC, IR)
            }
        } catch (e: Throwable) {
            checkException(sPC, e)
        } finally {
            cpu.totalExecuted += n
        }
        return n
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


@Suppress("FunctionName")
// http://www.mrc.uidaho.edu/mrc/people/jff/digital/MIPSir.html
class InstructionInterpreter(val s: CpuState) : InstructionEvaluator<CpuState>() {
    override fun unimplemented(s: CpuState, i: InstructionType): Unit =
        TODO("unimplemented: ${i.name} : " + i + " at ${"%08X".format(s._PC)}")

    private inline operator fun Int.invoke(callback: Int.() -> Unit) {
        callback()
        s.advance_pc(4)
    }

    //inline val mem: Memory get() = s.mem
    val mem = s.mem

    val itemp = IntArray(2)

    inline var Int.HI_LO: Long; get() = s.HI_LO; set(value) = run { s.HI_LO = value }
    inline var Int.LO: Int; get() = s.LO; set(value) = run { s.LO = value }
    inline var Int.HI: Int; get() = s.HI; set(value) = run { s.HI = value }
    inline var Int.IC: Int; get() = s.IC; set(value) = run { s.IC = value }

    inline var Int.RD: Int; get() = s.getGpr(this.rd); set(value) = run { s.setGpr(this.rd, value) }
    inline var Int.RT: Int; get() = s.getGpr(this.rt); set(value) = run { s.setGpr(this.rt, value) }
    inline var Int.RS: Int; get() = s.getGpr(this.rs); set(value) = run { s.setGpr(this.rs, value) }

    inline var Int.FD: Float; get() = s.getFpr(this.fd); set(value) = run { s.setFpr(this.fd, value) }
    inline var Int.FT: Float; get() = s.getFpr(this.ft); set(value) = run { s.setFpr(this.ft, value) }
    inline var Int.FS: Float; get() = s.getFpr(this.fs); set(value) = run { s.setFpr(this.fs, value) }

    inline var Int.FD_I: Int; get() = s.getFprI(this.fd); set(value) = run { s.setFprI(this.fd, value) }
    inline var Int.FT_I: Int; get() = s.getFprI(this.ft); set(value) = run { s.setFprI(this.ft, value) }
    inline var Int.FS_I: Int; get() = s.getFprI(this.fs); set(value) = run { s.setFprI(this.fs, value) }

    inline val Int.S_IMM14: Int; get() = this.s_imm14
    inline val Int.S_IMM16: Int; get() = this.s_imm16
    inline val Int.U_IMM16: Int; get() = this.u_imm16
    inline val Int.POS: Int get() = this.pos
    inline val Int.SIZE_E: Int get() = this.size_e
    inline val Int.SIZE_I: Int get() = this.size_i
    inline val Int.RS_IMM16: Int; get() = RS + S_IMM16
    inline val Int.RS_IMM14: Int; get() = RS + S_IMM14 * 4

    // ALU
    override fun lui(i: Int, s: CpuState) = i { RT = (U_IMM16 shl 16) }

    override fun movz(i: Int, s: CpuState) = i { RD = dyna_movz(RT, RD, RS) }
    override fun movn(i: Int, s: CpuState) = i { RD = dyna_movn(RT, RD, RS) }

    //override fun movz(i: Int, s: CpuState) = i { if (RT == 0) RD = RS }
    //override fun movn(i: Int, s: CpuState) = i { if (RT != 0) RD = RS }

    override fun ext(i: Int, s: CpuState) = i { RT = dyna_ext(RS, POS, SIZE_E) }
    override fun ins(i: Int, s: CpuState) = i { RT = dyna_ins(RT, RS, POS, SIZE_I) }

    //override fun ext(i: Int, s: CpuState) = i { RT = RS.extract(POS, SIZE_E) }
    //override fun ins(i: Int, s: CpuState) = i { RT = RT.insert(RS, POS, SIZE_I) }

    override fun clz(i: Int, s: CpuState) = i { RD = dyna_clz(RS) }
    override fun clo(i: Int, s: CpuState) = i { RD = dyna_clo(RS) }
    override fun seb(i: Int, s: CpuState) = i { RD = dyna_seb(RT) }
    override fun seh(i: Int, s: CpuState) = i { RD = dyna_seh(RT) }

    override fun wsbh(i: Int, s: CpuState) = i { RD = dyna_wsbh(RT) }
    override fun wsbw(i: Int, s: CpuState) = i { RD = dyna_wsbw(RT) }

    override fun max(i: Int, s: CpuState) = i { RD = dyna_max(RS, RT) }
    override fun min(i: Int, s: CpuState) = i { RD = dyna_min(RS, RT) }

    override fun add(i: Int, s: CpuState) = i { RD = RS + RT }
    override fun addu(i: Int, s: CpuState) = i { RD = RS + RT }
    override fun sub(i: Int, s: CpuState) = i { RD = RS - RT }
    override fun subu(i: Int, s: CpuState) = i { RD = RS - RT }
    override fun addi(i: Int, s: CpuState) = i { RT = RS + S_IMM16 }
    override fun addiu(i: Int, s: CpuState) = i { RT = RS + S_IMM16 }


    override fun div(i: Int, s: CpuState) = i { s.div(RS, RT) }
    override fun divu(i: Int, s: CpuState) = i { s.divu(RS, RT) }
    override fun mult(i: Int, s: CpuState) = i { s.mult(RS, RT) }
    override fun multu(i: Int, s: CpuState) = i { s.multu(RS, RT) }
    override fun madd(i: Int, s: CpuState) = i { s.madd(RS, RT) }
    override fun maddu(i: Int, s: CpuState) = i { s.maddu(RS, RT) }
    override fun msub(i: Int, s: CpuState) = i { s.msub(RS, RT) }
    override fun msubu(i: Int, s: CpuState) = i { s.msubu(RS, RT) }

    /*
    override fun div(i: Int, s: CpuState) = i { LO = RS / RT; HI = RS % RT }
    override fun divu(i: Int, s: CpuState) = i {
        val d = RT
        if (d != 0) {
            LO = RS udiv d
            HI = RS urem d
        } else {
            LO = 0
            HI = 0
        }
    }

   override fun mult(i: Int, s: CpuState) = i { imul32_64(RS, RT, itemp); this.LO = itemp[0]; this.HI = itemp[1] }
    override fun multu(i: Int, s: CpuState) = i { umul32_64(RS, RT, itemp); this.LO = itemp[0]; this.HI = itemp[1] }
    override fun madd(i: Int, s: CpuState) = i { HI_LO += RS.toLong() * RT.toLong() }
    override fun maddu(i: Int, s: CpuState) = i { HI_LO += RS.unsigned * RT.unsigned }
    override fun msub(i: Int, s: CpuState) = i { HI_LO -= RS.toLong() * RT.toLong() }
    override fun msubu(i: Int, s: CpuState) = i { HI_LO -= RS.unsigned * RT.unsigned }
    */


    override fun mflo(i: Int, s: CpuState) = i { RD = LO }
    override fun mfhi(i: Int, s: CpuState) = i { RD = HI }
    override fun mfic(i: Int, s: CpuState) = i { RT = IC }

    override fun mtlo(i: Int, s: CpuState) = i { LO = RS }
    override fun mthi(i: Int, s: CpuState) = i { HI = RS }
    override fun mtic(i: Int, s: CpuState) = i { IC = RT }

    // ALU: Bit
    override fun or(i: Int, s: CpuState) = i { RD = RS or RT }

    override fun xor(i: Int, s: CpuState) = i { RD = RS xor RT }
    override fun and(i: Int, s: CpuState) = i { RD = RS and RT }
    override fun nor(i: Int, s: CpuState) = i { RD = (RS or RT).inv() }

    override fun ori(i: Int, s: CpuState) = i { RT = RS or U_IMM16 }
    override fun xori(i: Int, s: CpuState) = i { RT = RS xor U_IMM16 }
    override fun andi(i: Int, s: CpuState) = i { RT = RS and U_IMM16 }

    override fun sll(i: Int, s: CpuState) = i { RD = dyna_sll(RT, POS) }
    override fun sra(i: Int, s: CpuState) = i { RD = dyna_sra(RT, POS) }
    override fun srl(i: Int, s: CpuState) = i { RD = dyna_srl(RT, POS) }
    override fun sllv(i: Int, s: CpuState) = i { RD = dyna_sll(RT, RS) }
    override fun srav(i: Int, s: CpuState) = i { RD = dyna_sra(RT, RS) }
    override fun srlv(i: Int, s: CpuState) = i { RD = dyna_srl(RT, RS) }

    //override fun sll(i: Int, s: CpuState) = i { RD = RT shl POS }
    //override fun sra(i: Int, s: CpuState) = i { RD = RT shr POS }
    //override fun srl(i: Int, s: CpuState) = i { RD = RT ushr POS }
    //override fun sllv(i: Int, s: CpuState) = i { RD = RT shl (RS and 0b11111) }
    //override fun srav(i: Int, s: CpuState) = i { RD = RT shr (RS and 0b11111) }
    //override fun srlv(i: Int, s: CpuState) = i { RD = RT ushr (RS and 0b11111) }

    override fun bitrev(i: Int, s: CpuState) = i { RD = BitUtils.bitrev32(RT) }

    override fun rotr(i: Int, s: CpuState) = i { RD = BitUtils.rotr(RT, POS) }
    override fun rotrv(i: Int, s: CpuState) = i { RD = BitUtils.rotr(RT, RS) }

    // Memory
    override fun lb(i: Int, s: CpuState) = i { RT = s.lb(RS_IMM16) }

    override fun lbu(i: Int, s: CpuState) = i { RT = s.lbu(RS_IMM16) }
    override fun lh(i: Int, s: CpuState) = i { RT = s.lh(RS_IMM16) }
    override fun lhu(i: Int, s: CpuState) = i { RT = s.lhu(RS_IMM16) }
    override fun lw(i: Int, s: CpuState) = i { RT = s.lw(RS_IMM16) }
    override fun ll(i: Int, s: CpuState) = i { RT = s.lw(RS_IMM16) }

    override fun lwl(i: Int, s: CpuState) = i { RT = s.lwl(RS_IMM16, RT) }
    override fun lwr(i: Int, s: CpuState) = i { RT = s.lwr(RS_IMM16, RT) }

    override fun swl(i: Int, s: CpuState) = i { s.swl(RS_IMM16, RT) }
    override fun swr(i: Int, s: CpuState) = i { s.swr(RS_IMM16, RT) }

    override fun sb(i: Int, s: CpuState) = i { s.sb(RS_IMM16, RT) }
    override fun sh(i: Int, s: CpuState) = i { s.sh(RS_IMM16, RT) }
    override fun sw(i: Int, s: CpuState) = i { s.sw(RS_IMM16, RT) }
    override fun sc(i: Int, s: CpuState) = i { s.sw(RS_IMM16, RT); RT = 1 }

    override fun lwc1(i: Int, s: CpuState) = i { FT_I = s.lw(RS_IMM16) }
    override fun swc1(i: Int, s: CpuState) = i { s.sw(RS_IMM16, FT_I) }

    // Special
    override fun syscall(i: Int, s: CpuState) = s.preadvance { syscall(SYSCALL) }

    override fun _break(i: Int, s: CpuState) = s.preadvance { throw CpuBreakExceptionCached(SYSCALL) }

    // Set less
    //override fun slt(i: Int, s: CpuState) = s { RD = (RS < RT).toInt() }
    //override fun sltu(i: Int, s: CpuState) = s { RD = (RS ult RT).toInt() }

    //override fun slti(i: Int, s: CpuState) = s { RT = (RS < S_IMM16).toInt() }
    //override fun sltiu(i: Int, s: CpuState) = s { RT = (RS ult S_IMM16).toInt() }

    override fun slt(i: Int, s: CpuState) = s { RD = s.slt(RS, RT) }
    override fun sltu(i: Int, s: CpuState) = s { RD = s.sltu(RS, RT) }
    override fun slti(i: Int, s: CpuState) = s { RT = s.slt(RS, S_IMM16) }
    override fun sltiu(i: Int, s: CpuState) = s { RT = s.sltu(RS, S_IMM16) }


    // Branch
    override fun beq(i: Int, s: CpuState) = s.branch { RS == RT }

    override fun bne(i: Int, s: CpuState) = s.branch { RS != RT }
    override fun bltz(i: Int, s: CpuState) = s.branch { RS < 0 }
    override fun blez(i: Int, s: CpuState) = s.branch { RS <= 0 }
    override fun bgtz(i: Int, s: CpuState) = s.branch { RS > 0 }
    override fun bgez(i: Int, s: CpuState) = s.branch { RS >= 0 }
    override fun bgezal(i: Int, s: CpuState) = s.branch { RA = _nPC + 4; RS >= 0 }
    override fun bltzal(i: Int, s: CpuState) = s.branch { RA = _nPC + 4; RS < 0 }

    override fun beql(i: Int, s: CpuState) = s.branchLikely { RS == RT }
    override fun bnel(i: Int, s: CpuState) = s.branchLikely { RS != RT }
    override fun bltzl(i: Int, s: CpuState) = s.branchLikely { RS < 0 }
    override fun blezl(i: Int, s: CpuState) = s.branchLikely { RS <= 0 }
    override fun bgtzl(i: Int, s: CpuState) = s.branchLikely { RS > 0 }
    override fun bgezl(i: Int, s: CpuState) = s.branchLikely { RS >= 0 }
    override fun bgezall(i: Int, s: CpuState) = s.branchLikely { RA = _nPC + 4; RS >= 0 }
    override fun bltzall(i: Int, s: CpuState) = s.branchLikely { RA = _nPC + 4; RS < 0 }


    override fun bc1f(i: Int, s: CpuState) = s.branch { !fcr31_cc }
    override fun bc1t(i: Int, s: CpuState) = s.branch { fcr31_cc }
    override fun bc1fl(i: Int, s: CpuState) = s.branchLikely { !fcr31_cc }
    override fun bc1tl(i: Int, s: CpuState) = s.branchLikely { fcr31_cc }

    //override fun j(i: Int, s: CpuState) = s.none { _PC = _nPC; _nPC = (_PC and 0xf0000000.toInt()) or (JUMP_ADDRESS) } // @TODO: Kotlin.JS doesn't optimize 0xf0000000.toInt() and generates a long
    override fun j(i: Int, s: CpuState) = s.none { _PC = _nPC; _nPC = (_PC and (-268435456)) or (JUMP_ADDRESS) }

    override fun jr(i: Int, s: CpuState) = s.none { _PC = _nPC; _nPC = RS }

    override fun jal(i: Int, s: CpuState) = s.none {
        j(i, s); RA = _PC + 4;
    } // $31 = PC + 8 (or nPC + 4); PC = nPC; nPC = (PC & 0xf0000000) | (target << 2);

    override fun jalr(i: Int, s: CpuState) = s.none { jr(i, s); RD = _PC + 4; }

    // Float
    override fun mfc1(i: Int, s: CpuState) = s { RT = FS_I }

    override fun mtc1(i: Int, s: CpuState) = s { FS_I = RT }
    override fun cvt_s_w(i: Int, s: CpuState) = s { FD = FS_I.toFloat() }
    override fun cvt_w_s(i: Int, s: CpuState) = s {

        FD_I = when (this.fcr31_rm) {
            0 -> Math.rint(FS) // rint: round nearest
            1 -> Math.cast(FS) // round to zero
            2 -> Math.ceil(FS) // round up (ceil)
            3 -> Math.floor(FS) // round down (floor)
            else -> FS.toInt()
        }
    }

    override fun trunc_w_s(i: Int, s: CpuState) = s { FD_I = Math.trunc(FS) }
    override fun round_w_s(i: Int, s: CpuState) = s { FD_I = Math.round(FS) }
    override fun ceil_w_s(i: Int, s: CpuState) = s { FD_I = Math.ceil(FS) }
    override fun floor_w_s(i: Int, s: CpuState) = s { FD_I = Math.floor(FS) }

    inline fun CpuState.checkNan(callback: CpuState.() -> Unit) = this.normal {
        callback()
        if (FD.isNaN()) fcr31 = fcr31 or 0x00010040
        if (FD.isInfinite()) fcr31 = fcr31 or 0x00005014
    }

    override fun mov_s(i: Int, s: CpuState) = s.checkNan { FD = FS }
    override fun add_s(i: Int, s: CpuState) = s.checkNan { FD = FS pspAdd FT }
    override fun sub_s(i: Int, s: CpuState) = s.checkNan { FD = FS pspSub FT }
    override fun mul_s(i: Int, s: CpuState) = s.checkNan { FD = FS * FT; if (fcr31_fs && FD.isAlmostZero()) FD = 0f }
    override fun div_s(i: Int, s: CpuState) = s.checkNan { FD = FS / FT }
    override fun neg_s(i: Int, s: CpuState) = s.checkNan { FD = -FS }
    override fun abs_s(i: Int, s: CpuState) = s.checkNan { FD = kotlin.math.abs(FS) }
    override fun sqrt_s(i: Int, s: CpuState) = s.checkNan { FD = kotlin.math.sqrt(FS) }

    private inline fun CpuState._cu(callback: CpuState.() -> Boolean) =
        this { fcr31_cc = if (FS.isNaN() || FT.isNaN()) true else callback() }

    private inline fun CpuState._co(callback: CpuState.() -> Boolean) =
        this { fcr31_cc = if (FS.isNaN() || FT.isNaN()) false else callback() }

    override fun c_f_s(i: Int, s: CpuState) = s._co { false }
    override fun c_un_s(i: Int, s: CpuState) = s._cu { false }
    override fun c_eq_s(i: Int, s: CpuState) = s._co { FS == FT }
    override fun c_ueq_s(i: Int, s: CpuState) = s._cu { FS == FT }
    override fun c_olt_s(i: Int, s: CpuState) = s._co { FS < FT }
    override fun c_ult_s(i: Int, s: CpuState) = s._cu { FS < FT }
    override fun c_ole_s(i: Int, s: CpuState) = s._co { FS <= FT }
    override fun c_ule_s(i: Int, s: CpuState) = s._cu { FS <= FT }

    override fun c_sf_s(i: Int, s: CpuState) = s._co { false }
    override fun c_ngle_s(i: Int, s: CpuState) = s._cu { false }
    override fun c_seq_s(i: Int, s: CpuState) = s._co { FS == FT }
    override fun c_ngl_s(i: Int, s: CpuState) = s._cu { FS == FT }
    override fun c_lt_s(i: Int, s: CpuState) = s._co { FS < FT }
    override fun c_nge_s(i: Int, s: CpuState) = s._cu { FS < FT }
    override fun c_le_s(i: Int, s: CpuState) = s._co { FS <= FT }
    override fun c_ngt_s(i: Int, s: CpuState) = s._cu { FS <= FT }

    override fun cfc1(i: Int, s: CpuState) = s {
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

    override fun ctc1(i: Int, s: CpuState) = s {
        when (IR.rd) {
            31 -> updateFCR31(RT)
        }
    }

    private val VDEST2 = IntArray2(4, 4)
    private val VSRC = IntArray(16)

    private fun _lv_x(s: CpuState, size: Int) = s {
        getVectorRegisters(VSRC, IR.vt5_1, size)
        val start = RS_IMM14
        for (n in 0 until size) s.VFPRI[VSRC[n]] = mem.lw(start + n * 4)
    }

    private fun _sv_x(s: CpuState, size: Int) = s {
        getVectorRegisters(VSRC, IR.vt5_1, size)
        val start = RS_IMM14
        for (n in 0 until size) mem.sw(start + n * 4, s.VFPRI[VSRC[n]])
    }

    override fun lv_s(i: Int, s: CpuState) = _lv_x(s, 1)
    override fun lv_q(i: Int, s: CpuState) = _lv_x(s, 4)

    override fun sv_s(i: Int, s: CpuState) = _sv_x(s, 1)
    override fun sv_q(i: Int, s: CpuState) = _sv_x(s, 4)

    override fun lvl_q(i: Int, s: CpuState) = s {
        getVectorRegisters(VSRC, IR.vt5_1, 4)
        mem.lvl_q(RS_IMM14) { i, value -> s.setVfprI(VSRC[i], value) }
    }

    override fun lvr_q(i: Int, s: CpuState) = s {
        getVectorRegisters(VSRC, IR.vt5_1, 4)
        mem.lvr_q(RS_IMM14) { i, value -> s.setVfprI(VSRC[i], value) }
    }

    override fun svl_q(i: Int, s: CpuState) = s {
        getVectorRegisters(VSRC, IR.vt5_1, 4)
        mem.svl_q(RS_IMM14) { getVfprI(VSRC[it]) }
    }

    override fun svr_q(i: Int, s: CpuState) = s {
        getVectorRegisters(VSRC, IR.vt5_1, 4)
        mem.svr_q(RS_IMM14) { getVfprI(VSRC[it]) }
    }

    private fun cc_8888_to_4444(i: Int): Int = 0 or
            (((i ushr 4) and 15) shl 0) or
            (((i ushr 12) and 15) shl 4) or
            (((i ushr 20) and 15) shl 8) or
            (((i ushr 28) and 15) shl 12)

    private fun cc_8888_to_5551(i: Int): Int = 0 or
            (((i ushr 3) and 31) shl 0) or
            (((i ushr 11) and 31) shl 5) or
            (((i ushr 19) and 31) shl 10) or
            (((i ushr 31) and 1) shl 15)

    private fun cc_8888_to_5650(i: Int): Int = 0 or
            (((i ushr 3) and 31) shl 0) or
            (((i ushr 10) and 63) shl 5) or
            (((i ushr 19) and 31) shl 11)

    private fun CpuState._vtXXXX_q(func: (Int) -> Int) = this {
        setVDI_VS(destSize = IR.one_two / 2) {
            func(vsi[it * 2 + 0]) or (func(vsi[it * 2 + 1]) shl 16)
        }
    }

    override fun vt4444_q(i: Int, s: CpuState) = s._vtXXXX_q(this::cc_8888_to_4444)
    override fun vt5551_q(i: Int, s: CpuState) = s._vtXXXX_q(this::cc_8888_to_5551)
    override fun vt5650_q(i: Int, s: CpuState) = s._vtXXXX_q(this::cc_8888_to_5650)

    private fun _vc2i(s: CpuState, func: (index: Int, value: Int) -> Int) = s {
        setVDI_VS(destSize = 4, srcSize = 1) { func(it, vsi.x) }
    }

    override fun vc2i(i: Int, s: CpuState) =
        _vc2i(s) { index, value -> (value shl ((3 - index) * 8)) and 0xFF000000.toInt() }

    override fun vuc2i(i: Int, s: CpuState) =
        _vc2i(s) { index, value -> ((((value ushr (index * 8)) and 0xFF) * 0x01010101) shr 1) and 0x80000000.toInt().inv() }

    private fun _vs2i(s: CpuState, func: (index: Int, value: Int) -> Int) = s {
        setVDI_VS(destSize = IR.one_two * 2) { func(it % 2, vsi[it / 2]) }
    }

    override fun vs2i(i: Int, s: CpuState) = _vs2i(s) { index, value -> value.extract(index * 16, 16) shl 16 }
    override fun vus2i(i: Int, s: CpuState) = _vs2i(s) { index, value -> value.extract(index * 16, 16) shl 15 }

    private fun _vi2c(s: CpuState, gen: (value: Int) -> Int) = s {
        setVDI_VS(destSize = 1, srcSize = 4) {
            RGBA.packFast(gen(vsi[0]), gen(vsi[1]), gen(vsi[2]), gen(vsi[3]))
        }
    }

    override fun vi2c(i: Int, s: CpuState) = _vi2c(s) { it.extract8(24) }
    override fun vi2uc(i: Int, s: CpuState) = _vi2c(s) { if (it < 0) 0 else it.extract8(23) }

    private fun _vi2s(s: CpuState, gen: (value: Int) -> Int) = s {
        setVDI_VS(destSize = IR.one_two / 2) {
            val l = gen(vsi[it * 2 + 0])
            val r = gen(vsi[it * 2 + 1])
            l or (r shl 16)
        }
    }

    override fun vi2s(i: Int, s: CpuState) = _vi2s(s) { it ushr 16 }
    override fun vi2us(i: Int, s: CpuState) = _vi2s(s) { if (it < 0) 0 else it shr 15 }
    override fun vi2f(i: Int, s: CpuState) = s { setVD_VS { vsi[it] * 2f.pow(-IR.imm5) } }

    private fun _vf2ix(s: CpuState, func: (value: Float, imm5: Int) -> Int) = s {
        setVDI_VS { if (vs[it].isNaN()) 0x7FFFFFFF else func(vs[it], IR.imm5) }
    }

    override fun vf2id(i: Int, s: CpuState) = _vf2ix(s) { value, imm5 -> floor(value * 2f.pow(imm5)).toInt() }
    override fun vf2iu(i: Int, s: CpuState) = _vf2ix(s) { value, imm5 -> ceil(value * 2f.pow(imm5)).toInt() }
    override fun vf2in(i: Int, s: CpuState) = _vf2ix(s) { value, imm5 -> Math.rint((value * 2f.pow(imm5))) }
    override fun vf2iz(i: Int, s: CpuState) = _vf2ix(s) { value, imm5 ->
        val rs = value * 2f.pow(imm5); if (value >= 0) floor(rs).toInt() else ceil(rs).toInt()
    }

    override fun vf2h(i: Int, s: CpuState) = s {
        setVDI_VS(destSize = IR.one_two / 2) {
            val l = HalfFloat.floatBitsToHalfFloatBits(vsi[it * 2 + 0])
            val r = HalfFloat.floatBitsToHalfFloatBits(vsi[it * 2 + 1])
            (l) or (r shl 16)
        }
    }

    override fun vh2f(i: Int, s: CpuState) = s {
        setVDI_VS(destSize = IR.one_two * 2) {
            HalfFloat.halfFloatBitsToFloatBits(vsi[it / 2].extract((it % 2) * 16, 16))
        }
    }

    override fun viim(i: Int, s: CpuState) = s { VT = S_IMM16.toFloat() }
    override fun vfim(i: Int, s: CpuState) = s { VT_I = HalfFloat.halfFloatBitsToFloatBits(U_IMM16) }

    override fun vcst(i: Int, s: CpuState) = s { VD = VfpuConstants[IR.imm5].value }
    override fun mtv(i: Int, s: CpuState) = s { VD_I = RT }
    override fun vpfxt(i: Int, s: CpuState) = s { vpfxt.setEnable(IR) }
    override fun vpfxd(i: Int, s: CpuState) = s { vpfxd.setEnable(IR) }
    override fun vpfxs(i: Int, s: CpuState) = s { vpfxs.setEnable(IR) }
    override fun vavg(i: Int, s: CpuState) = s {
        setVD_VS(destSize = 1, prefixes = true) {
            ((0 until vsSize).sumByFloat { (vs[it] / vsSize) })
        }
    }

    override fun vfad(i: Int, s: CpuState) = s {
        setVD_VS(destSize = 1, prefixes = true) {
            ((0 until vsSize).sumByFloat { vs[it] })
        }
    }

    override fun vrot(i: Int, s: CpuState) = s {
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
    override fun vzero(i: Int, s: CpuState) = s { setVD_(prefixes = true) { 0f } }

    override fun vone(i: Int, s: CpuState) = s { setVD_(prefixes = true) { 1f } }

    // Vector operations (one operand)
    override fun vmov(i: Int, s: CpuState) = s { setVD_VS(prefixes = true) { vs[it] } }

    override fun vabs(i: Int, s: CpuState) = s { setVD_VS(prefixes = true) { abs(vs[it]) } }
    override fun vsqrt(i: Int, s: CpuState) = s { setVD_VS(prefixes = true) { sqrt(vs[it]) } }
    override fun vneg(i: Int, s: CpuState) = s { setVD_VS(prefixes = true) { -vs[it] } }
    override fun vsat0(i: Int, s: CpuState) = s { setVD_VS(prefixes = true) { vs[it].pspSat0 } }
    override fun vsat1(i: Int, s: CpuState) = s { setVD_VS(prefixes = true) { vs[it].pspSat1 } }
    override fun vrcp(i: Int, s: CpuState) = s { setVD_VS(prefixes = true) { 1f / vs[it] } }
    override fun vrsq(i: Int, s: CpuState) = s { setVD_VS(prefixes = true) { 1f / sqrt(vs[it]) } }
    override fun vsin(i: Int, s: CpuState) = s { setVD_VS(prefixes = true) { sinv1(vs[it]) } }
    override fun vasin(i: Int, s: CpuState) = s { setVD_VS(prefixes = true) { asinv1(vs[it]) } }
    override fun vnsin(i: Int, s: CpuState) = s { setVD_VS(prefixes = true) { -sinv1(vs[it]) } }
    override fun vcos(i: Int, s: CpuState) = s { setVD_VS(prefixes = true) { cosv1(vs[it]) } }
    override fun vexp2(i: Int, s: CpuState) = s { setVD_VS(prefixes = true) { 2f.pow(vs[it]) } }
    override fun vrexp2(i: Int, s: CpuState) = s { setVD_VS(prefixes = true) { 1f / 2f.pow(vs[it]) } }
    override fun vlog2(i: Int, s: CpuState) = s { setVD_VS(prefixes = true) { log2(vs[it]) } }
    override fun vnrcp(i: Int, s: CpuState) = s { setVD_VS(prefixes = true) { -1f / vs[it] } }
    override fun vsgn(i: Int, s: CpuState) = s { setVD_VS(prefixes = true) { vs[it].pspSign } }
    override fun vocp(i: Int, s: CpuState) = s { setVD_VS(prefixes = true) { 1f - vs[it] } }
    override fun vbfy1(i: Int, s: CpuState) = s {
        setVD_VS(prefixes = true) {
            when (it) {
                0 -> vs.x + vs.y
                1 -> vs.x - vs.y
                2 -> vs.z + vs.w
                3 -> vs.z - vs.w
                else -> invalidOp1
            }
        }
    }

    override fun vbfy2(i: Int, s: CpuState) = s {
        setVD_VS(prefixes = true) {
            when (it) {
                0 -> vs.x + vs.z
                1 -> vs.y + vs.w
                2 -> vs.x - vs.z
                3 -> vs.y - vs.w
                else -> invalidOp1
            }
        }
    }

    override fun vsrt1(i: Int, s: CpuState) = s {
        setVD_VS(prefixes = true) {
            when (it) {
                0 -> min(vs.x, vs.y)
                1 -> max(vs.x, vs.y)
                2 -> min(vs.z, vs.w)
                3 -> max(vs.z, vs.w)
                else -> invalidOp1
            }
        }
    }

    override fun vsrt2(i: Int, s: CpuState) = s {
        setVD_VS(prefixes = true) {
            vs.run {
                when (it) {
                    0 -> min(x, w)
                    1 -> min(y, z)
                    2 -> max(y, z)
                    3 -> max(x, w)
                    else -> invalidOp1
                }
            }
        }
    }

    override fun vsrt3(i: Int, s: CpuState) = s {
        setVD_VS(prefixes = true) {
            when (it) {
                0 -> max(vs.x, vs.y)
                1 -> min(vs.x, vs.y)
                2 -> max(vs.z, vs.w)
                3 -> min(vs.z, vs.w)
                else -> invalidOp1
            }
        }
    }

    override fun vsrt4(i: Int, s: CpuState) = s {
        setVD_VS(prefixes = true) {
            when (it) {
                0 -> max(vs.x, vs.w)
                1 -> max(vs.y, vs.z)
                2 -> min(vs.y, vs.z)
                3 -> min(vs.x, vs.w)
                else -> invalidOp1
            }
        }
    }

    // Vector operations (two operands)
    override fun vsge(i: Int, s: CpuState) = s { setVD_VSVT(prefixes = true) { if (vs[it] >= vt[it]) 1f else 0f } }

    override fun vslt(i: Int, s: CpuState) = s { setVD_VSVT(prefixes = true) { if (vs[it] < vt[it]) 1f else 0f } }
    override fun vscmp(i: Int, s: CpuState) = s { setVD_VSVT(prefixes = true) { vs[it].compareTo(vt[it]).toFloat() } }

    override fun vadd(i: Int, s: CpuState) = s { setVD_VSVT(prefixes = true) { vs[it] pspAdd vt[it] } }
    override fun vsub(i: Int, s: CpuState) = s { setVD_VSVT(prefixes = true) { vs[it] pspSub vt[it] } }
    override fun vmul(i: Int, s: CpuState) = s { setVD_VSVT(prefixes = true) { vs[it] * vt[it] } }
    override fun vdiv(i: Int, s: CpuState) = s { setVD_VSVT(prefixes = true) { vs[it] / vt[it] } }
    override fun vmin(i: Int, s: CpuState) = s { setVD_VSVT(prefixes = true) { min(vs[it], vt[it]) } }
    override fun vmax(i: Int, s: CpuState) = s { setVD_VSVT(prefixes = true) { max(vs[it], vt[it]) } }
    override fun vcrs_t(i: Int, s: CpuState) = s {
        setVD_VSVT(prefixes = true) {
            when (it) {
                0 -> vs.y * vt.z
                1 -> vs.z * vt.x
                2 -> vs.x * vt.y
                else -> invalidOp1
            }
        }
    }

    override fun vcrsp_t(i: Int, s: CpuState) = s {
        setVD_VSVT(prefixes = true) {
            when (it) {
                0 -> +vs.y * vt.z - vs.z * vt.y
                1 -> +vs.z * vt.x - vs.x * vt.z
                2 -> +vs.x * vt.y - vs.y * vt.x
                else -> invalidOp1
            }
        }
    }

    override fun vqmul(i: Int, s: CpuState) = s {
        setVD_VSVT(prefixes = true) {
            when (it) {
                0 -> +vs.x * vt.w + vs.y * vt.z - vs.z * vt.y + vs.w * vt.x
                1 -> -vs.x * vt.z + vs.y * vt.w + vs.z * vt.x + vs.w * vt.y
                2 -> +vs.x * vt.y - vs.y * vt.x + vs.z * vt.w + vs.w * vt.z
                3 -> -vs.x * vt.x - vs.y * vt.y - vs.z * vt.z + vs.w * vt.w
                else -> invalidOp1
            }
        }
    }

    override fun vdot(i: Int, s: CpuState) = s {
        setVD_VSVT(destSize = 1, prefixes = true) {
            ((0 until vsSize).sumByFloat { (vs[it] * vt[it]) })
        }
    }

    override fun vscl(i: Int, s: CpuState) = s { setVD_VSVT(targetSize = 1, prefixes = true) { vs[it] * vt.x } }

    override fun vhdp(i: Int, s: CpuState) = s {
        setVD_VSVT(destSize = 1, prefixes = true) {
            vs[vsSize - 1] = 1f
            (0 until vsSize).sumByFloat { (vs[it] * vt[it]) }
        }
    }

    override fun vdet(i: Int, s: CpuState) = s {
        setVD_VSVT(destSize = 1, prefixes = true) {
            vs.x * vt.y - vs.y * vt.x
        }
    }

    override fun vcmp(i: Int, s: CpuState) = s {
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

    private fun _vcmovtf(s: CpuState, truth: Boolean) = s {
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

    override fun vcmovf(i: Int, s: CpuState) = _vcmovtf(s, true)
    override fun vcmovt(i: Int, s: CpuState) = _vcmovtf(s, false)

    override fun vwbn(i: Int, s: CpuState) = s {
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

    override fun vsbn(i: Int, s: CpuState) = s {
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
    override fun vmzero(i: Int, s: CpuState) = s { setMatrixVD { 0f } }

    override fun vmone(i: Int, s: CpuState) = s { setMatrixVD { 1f } }
    override fun vmidt(i: Int, s: CpuState) = s { setMatrixVD { if (row == col) 1f else 0f } }
    override fun vmmov(i: Int, s: CpuState) = s { setMatrixVD_VS { ms[col, row] } }
    override fun vmmul(i: Int, s: CpuState) = s {
        setMatrixVD_VSVT {
            (0 until side).map { ms[row, it] * mt[col, it] }.sum()
        }
    }

    override fun mfvc(i: Int, s: CpuState) = s { RT = VFPRC[IR.imm7] }
    override fun mtvc(i: Int, s: CpuState) = s { VFPRC[IR.imm7] = RT }

    private fun _vtfm_x(s: CpuState, size: Int) = s {
        vfpuContext.run {
            getVectorRegisterValues(b_vt, IR.vt, size)

            for (n in 0 until size) {
                getVectorRegisterValues(b_vs, IR.vs + n, size)
                vfpuContext.vd[n] = (0 until size).sumByFloat { (vs[it] * vt[it]) }
            }

            setVectorRegisterValues(b_vd, IR.vd, size)
        }
    }

    private fun _vhtfm_x(s: CpuState, size: Int) = s {
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

    override fun vtfm2(i: Int, s: CpuState) = _vtfm_x(s, 2)
    override fun vtfm3(i: Int, s: CpuState) = _vtfm_x(s, 3)
    override fun vtfm4(i: Int, s: CpuState) = _vtfm_x(s, 4)

    override fun vhtfm2(i: Int, s: CpuState) = _vhtfm_x(s, 2)
    override fun vhtfm3(i: Int, s: CpuState) = _vhtfm_x(s, 3)
    override fun vhtfm4(i: Int, s: CpuState) = _vhtfm_x(s, 4)

    override fun vmscl(i: Int, s: CpuState) = s {
        val scale = vfpuContext.sreadVt(s, size = 1)[0]
        setMatrixVD_VS { ms[col, row] * scale }
    }

    override fun vidt(i: Int, s: CpuState) = unimplemented(s, Instructions.vidt)
    override fun vnop(i: Int, s: CpuState) = unimplemented(s, Instructions.vnop)
    override fun vsync(i: Int, s: CpuState) = unimplemented(s, Instructions.vsync)
    override fun vflush(i: Int, s: CpuState) = unimplemented(s, Instructions.vflush)
    override fun vrnds(i: Int, s: CpuState) = unimplemented(s, Instructions.vrnds)
    override fun vrndi(i: Int, s: CpuState) = unimplemented(s, Instructions.vrndi)
    override fun vrndf1(i: Int, s: CpuState) = unimplemented(s, Instructions.vrndf1)
    override fun vrndf2(i: Int, s: CpuState) = unimplemented(s, Instructions.vrndf2)
    override fun vmfvc(i: Int, s: CpuState) = unimplemented(s, Instructions.vmfvc)
    override fun vmtvc(i: Int, s: CpuState) = unimplemented(s, Instructions.vmtvc)
    override fun mfvme(i: Int, s: CpuState) = unimplemented(s, Instructions.mfvme)
    override fun mtvme(i: Int, s: CpuState) = unimplemented(s, Instructions.mtvme)
    override fun vlgb(i: Int, s: CpuState) = unimplemented(s, Instructions.vlgb)
    override fun vsbz(i: Int, s: CpuState) = unimplemented(s, Instructions.vsbz)
    override fun vsocp(i: Int, s: CpuState) = unimplemented(s, Instructions.vsocp)
    override fun bvf(i: Int, s: CpuState) = unimplemented(s, Instructions.bvf)
    override fun bvt(i: Int, s: CpuState) = unimplemented(s, Instructions.bvt)
    override fun bvfl(i: Int, s: CpuState) = unimplemented(s, Instructions.bvfl)
    override fun bvtl(i: Int, s: CpuState) = unimplemented(s, Instructions.bvtl)

    // Vectorial utilities

    fun CpuState.getMatrixRegsValues(out: FloatArray2, matrixReg: Int, N: Int): Int {
        val side = getMatrixRegs(tempRegs2, matrixReg, N)
        for (j in 0 until side) for (i in 0 until side) out[j, i] = getVfpr(tempRegs2[j, i])
        return side
    }

    fun CpuState.setMatrixRegsValues(inp: FloatArray2, matrixReg: Int, N: Int): Int {
        val side = getMatrixRegs(tempRegs2, matrixReg, N)
        for (j in 0 until side) for (i in 0 until side) setVfpr(tempRegs2[j, i], inp[j, i])
        return side
    }

    fun getMatrixRegs(out: IntArray2, matrixReg: Int, N: Int): Int {
        val side = N
        val mtx = (matrixReg ushr 2) and 7
        val col = matrixReg and 3
        val transpose = ((matrixReg ushr 5) and 1) != 0
        val row = when (N) {
            2 -> (matrixReg ushr 5) and 2
            3 -> (matrixReg ushr 6) and 1
            4 -> (matrixReg ushr 5) and 2
            else -> invalidOp1
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

    private val mc = MatrixContext()

    fun CpuState.setMatrixVD(side: Int = IR.one_two, callback: MatrixContext.() -> Float) {
        mc.side = side
        for (col in 0 until side) for (row in 0 until side) {
            mc.md[col, row] = callback(mc.setPos(col, row))
        }
        setMatrixRegsValues(mc.md, IR.vd, side)
    }

    fun CpuState.setMatrixVD_VS(side: Int = IR.one_two, callback: MatrixContext.() -> Float) {
        getMatrixRegsValues(mc.ms, IR.vs, side)

        mc.side = side
        for (col in 0 until side) for (row in 0 until side) {
            mc.md[col, row] = callback(mc.setPos(col, row))
        }
        setMatrixRegsValues(mc.md, IR.vd, side)
    }

    fun CpuState.setMatrixVD_VSVT(side: Int = IR.one_two, callback: MatrixContext.() -> Float) {
        getMatrixRegsValues(mc.ms, IR.vs, side)
        getMatrixRegsValues(mc.mt, IR.vt, side)

        mc.side = side
        for (col in 0 until side) for (row in 0 until side) {
            mc.md[col, row] = callback(mc.setPos(col, row))
        }
        setMatrixRegsValues(mc.md, IR.vd, side)
    }

    private val tempRegs = IntArray(16)
    private val tempRegs2 = IntArray2(4, 4)

    fun getVectorRegisters(out: IntArray, vectorReg: Int, N: Int): Int {
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
            else -> invalidOp1
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

    fun CpuState.getVectorRegisterValues(out: FloatIntBuffer, vectorReg: Int, N: Int) {
        getVectorRegisters(tempRegs, vectorReg, N)
        for (n in 0 until N) out.i[n] = getVfprI(tempRegs[n])
    }

    fun CpuState.setVectorRegisterValues(inp: FloatIntBuffer, vectorReg: Int, N: Int) {
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

    inline fun CpuState.setVD_VS(
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

    override fun cache(i: Int, s: CpuState) = unimplemented(s, Instructions.cache)
    override fun sync(i: Int, s: CpuState) = unimplemented(s, Instructions.sync)
    override fun dbreak(i: Int, s: CpuState) = unimplemented(s, Instructions.dbreak)
    override fun halt(i: Int, s: CpuState) = unimplemented(s, Instructions.halt)
    override fun dret(i: Int, s: CpuState) = unimplemented(s, Instructions.dret)
    override fun eret(i: Int, s: CpuState) = unimplemented(s, Instructions.eret)
    override fun mfdr(i: Int, s: CpuState) = unimplemented(s, Instructions.mfdr)
    override fun mtdr(i: Int, s: CpuState) = unimplemented(s, Instructions.mtdr)
    override fun cfc0(i: Int, s: CpuState) = unimplemented(s, Instructions.cfc0)
    override fun ctc0(i: Int, s: CpuState) = unimplemented(s, Instructions.ctc0)
    override fun mfc0(i: Int, s: CpuState) = unimplemented(s, Instructions.mfc0)
    override fun mtc0(i: Int, s: CpuState) = unimplemented(s, Instructions.mtc0)
    override fun mfv(i: Int, s: CpuState) = unimplemented(s, Instructions.mfv)
}

