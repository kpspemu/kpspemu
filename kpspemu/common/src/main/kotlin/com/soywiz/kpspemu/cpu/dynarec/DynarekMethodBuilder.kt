package com.soywiz.kpspemu.cpu.dynarec

import com.soywiz.dynarek.*
import com.soywiz.kpspemu.cpu.*

class DynarekMethodBuilder : BaseDynarecMethodBuilder() {

    // ALU
    override fun lui(i: Int, s: InstructionInfo) = s { RT = ((IR.u_imm16 shl 16).lit) }

    override fun movz(i: Int, s: InstructionInfo) = s { RD = CpuState::movz.invoke(p0, RT, RD, RS) }
    override fun movn(i: Int, s: InstructionInfo) = s { RD = CpuState::movn.invoke(p0, RT, RD, RS) }

    override fun ext(i: Int, s: InstructionInfo) = s { RT = CpuState::ext.invoke(p0, RS, POS, SIZE_E) }
    override fun ins(i: Int, s: InstructionInfo) = s { RT = CpuState::ins.invoke(p0, RT, RS, POS, SIZE_I) }

    override fun clz(i: Int, s: InstructionInfo) = s { RD = CpuState::clz.invoke(p0, RS) }
    override fun clo(i: Int, s: InstructionInfo) = s { RD = CpuState::clo.invoke(p0, RS) }
    override fun seb(i: Int, s: InstructionInfo) = s { RD = CpuState::seb.invoke(p0, RT) }
    override fun seh(i: Int, s: InstructionInfo) = s { RD = CpuState::seh.invoke(p0, RT) }

    override fun wsbh(i: Int, s: InstructionInfo) = s { RD = CpuState::wsbh.invoke(p0, RT) }
    override fun wsbw(i: Int, s: InstructionInfo) = s { RD = CpuState::wsbw.invoke(p0, RT) }

    override fun max(i: Int, s: InstructionInfo) = s { RD = CpuState::max.invoke(p0, RS, RT) }
    override fun min(i: Int, s: InstructionInfo) = s { RD = CpuState::min.invoke(p0, RS, RT) }

    override fun add(i: Int, s: InstructionInfo) = s { RD = CpuState::add.invoke(p0, RS, RT) }
    override fun addu(i: Int, s: InstructionInfo) = s { RD = CpuState::add.invoke(p0, RS, RT) }
    override fun sub(i: Int, s: InstructionInfo) = s { RD = CpuState::sub.invoke(p0, RS, RT) }
    override fun subu(i: Int, s: InstructionInfo) = s { RD = CpuState::sub.invoke(p0, RS, RT) }
    override fun addi(i: Int, s: InstructionInfo) = s { RT = CpuState::add.invoke(p0, RS, S_IMM16) }
    override fun addiu(i: Int, s: InstructionInfo) = s { RT = CpuState::add.invoke(p0, RS, S_IMM16) }

    override fun div(i: Int, s: InstructionInfo) = s { STM(CpuState::div.invoke(p0, RS, RT)) }
    override fun divu(i: Int, s: InstructionInfo) = s { STM(CpuState::divu.invoke(p0, RS, RT)) }

    override fun mult(i: Int, s: InstructionInfo) = s { STM(CpuState::mult.invoke(p0, RS, RT)) }
    override fun multu(i: Int, s: InstructionInfo) = s { STM(CpuState::multu.invoke(p0, RS, RT)) }
    override fun madd(i: Int, s: InstructionInfo) = s { STM(CpuState::madd.invoke(p0, RS, RT)) }
    override fun maddu(i: Int, s: InstructionInfo) = s { STM(CpuState::maddu.invoke(p0, RS, RT)) }
    override fun msub(i: Int, s: InstructionInfo) = s { STM(CpuState::msub.invoke(p0, RS, RT)) }
    override fun msubu(i: Int, s: InstructionInfo) = s { STM(CpuState::msubu.invoke(p0, RS, RT)) }

    override fun mflo(i: Int, s: InstructionInfo) = s { RD = LO }
    override fun mfhi(i: Int, s: InstructionInfo) = s { RD = HI }
    override fun mfic(i: Int, s: InstructionInfo) = s { RT = IC }

    override fun mtlo(i: Int, s: InstructionInfo) = s { LO = RS }
    override fun mthi(i: Int, s: InstructionInfo) = s { HI = RS }
    override fun mtic(i: Int, s: InstructionInfo) = s { IC = RT }

    // ALU: Bit
    override fun or(i: Int, s: InstructionInfo) = s { RD = CpuState::or.invoke(p0, RS, RT) }
    override fun xor(i: Int, s: InstructionInfo) = s { RD = CpuState::xor.invoke(p0, RS, RT) }
    override fun and(i: Int, s: InstructionInfo) = s { RD = CpuState::and.invoke(p0, RS, RT) }
    override fun nor(i: Int, s: InstructionInfo) = s { RD = CpuState::nor.invoke(p0, RS, RT) }
    override fun ori(i: Int, s: InstructionInfo) = s { RT = CpuState::or.invoke(p0, RS, U_IMM16) }
    override fun xori(i: Int, s: InstructionInfo) = s { RT = CpuState::xor.invoke(p0, RS, U_IMM16) }
    override fun andi(i: Int, s: InstructionInfo) = s { RT = CpuState::and.invoke(p0, RS, U_IMM16) }

    override fun sll(i: Int, s: InstructionInfo) = s { RD = CpuState::sll.invoke(p0, RT, POS) }
    override fun sra(i: Int, s: InstructionInfo) = s { RD = CpuState::sra.invoke(p0, RT, POS) }
    override fun srl(i: Int, s: InstructionInfo) = s { RD = CpuState::srl.invoke(p0, RT, POS) }
    override fun sllv(i: Int, s: InstructionInfo) = s { RD = CpuState::sll.invoke(p0, RT, RS) }
    override fun srav(i: Int, s: InstructionInfo) = s { RD = CpuState::sra.invoke(p0, RT, RS) }
    override fun srlv(i: Int, s: InstructionInfo) = s { RD = CpuState::srl.invoke(p0, RT, RS) }
    override fun bitrev(i: Int, s: InstructionInfo) = s { RD = CpuState::bitrev32.invoke(p0, RT) }
    override fun rotr(i: Int, s: InstructionInfo) = s { RD = CpuState::rotr.invoke(p0, RT, POS) }
    override fun rotrv(i: Int, s: InstructionInfo) = s { RD = CpuState::rotr.invoke(p0, RT, RS) }

    // Memory
    override fun lb(i: Int, s: InstructionInfo) = s { RT = CpuState::lb.invoke(p0, RS_IMM16) }

    override fun lbu(i: Int, s: InstructionInfo) = s { RT = CpuState::lbu.invoke(p0, RS_IMM16) }
    override fun lh(i: Int, s: InstructionInfo) = s { RT = CpuState::lh.invoke(p0, RS_IMM16) }
    override fun lhu(i: Int, s: InstructionInfo) = s { RT = CpuState::lhu.invoke(p0, RS_IMM16) }
    override fun lw(i: Int, s: InstructionInfo) = s { RT = CpuState::lw.invoke(p0, RS_IMM16) }
    override fun ll(i: Int, s: InstructionInfo) = s { RT = CpuState::lw.invoke(p0, RS_IMM16) }

    override fun lwl(i: Int, s: InstructionInfo) = s { RT = CpuState::lwl.invoke(p0, RS_IMM16, RT) }
    override fun lwr(i: Int, s: InstructionInfo) = s { RT = CpuState::lwr.invoke(p0, RS_IMM16, RT) }

    override fun swl(i: Int, s: InstructionInfo) = s { STM(CpuState::swl.invoke(p0, RS_IMM16, RT)) }
    override fun swr(i: Int, s: InstructionInfo) = s { STM(CpuState::swr.invoke(p0, RS_IMM16, RT)) }

    override fun sb(i: Int, s: InstructionInfo) = s { STM(CpuState::sb.invoke(p0, RS_IMM16, RT)) }
    override fun sh(i: Int, s: InstructionInfo) = s { STM(CpuState::sh.invoke(p0, RS_IMM16, RT)) }
    override fun sw(i: Int, s: InstructionInfo) = s { STM(CpuState::sw.invoke(p0, RS_IMM16, RT)) }
    override fun sc(i: Int, s: InstructionInfo) = s { STM(CpuState::sw.invoke(p0, RS_IMM16, RT)); RT = 1.lit }

    override fun lwc1(i: Int, s: InstructionInfo) = s { FT_I = CpuState::lw.invoke(p0, RS_IMM16) }
    override fun swc1(i: Int, s: InstructionInfo) = s { STM(CpuState::sw.invoke(p0, RS_IMM16, FT_I)) }

    // Special
    override fun syscall(i: Int, s: InstructionInfo) =
        s.preadvanceAndFlow { STM(CpuState::syscall.invoke(p0, SYSCALL)) }

    override fun _break(i: Int, s: InstructionInfo) = s.preadvanceAndFlow { STM(CpuState::_break.invoke(p0, SYSCALL)) }

    // Set less
    override fun slt(i: Int, s: InstructionInfo) = s { RD = CpuState::slt.invoke(p0, RS, RT) }

    override fun sltu(i: Int, s: InstructionInfo) = s { RD = CpuState::sltu.invoke(p0, RS, RT) }
    override fun slti(i: Int, s: InstructionInfo) = s { RT = CpuState::slt.invoke(p0, RS, S_IMM16) }
    override fun sltiu(i: Int, s: InstructionInfo) = s { RT = CpuState::sltu.invoke(p0, RS, S_IMM16) }

    // Branch
    override fun beq(i: Int, s: InstructionInfo) = s.branch { RS eq RT }

    override fun bne(i: Int, s: InstructionInfo) = s.branch { RS ne RT }
    override fun bltz(i: Int, s: InstructionInfo) = s.branch { RS lt 0.lit }
    override fun blez(i: Int, s: InstructionInfo) = s.branch { RS le 0.lit }
    override fun bgtz(i: Int, s: InstructionInfo) = s.branch { RS gt 0.lit }
    override fun bgez(i: Int, s: InstructionInfo) = s.branch { RS ge 0.lit }
    //override fun bgezal(i: Int, s: InstructionInfo) = s.branch { RA = _nPC + 4; RS ge 0.lit }
    //override fun bltzal(i: Int, s: InstructionInfo) = s.branch { RA = _nPC + 4; RS lt 0.lit }

    override fun beql(i: Int, s: InstructionInfo) = s.branchLikely { RS eq RT }
    override fun bnel(i: Int, s: InstructionInfo) = s.branchLikely { RS ne RT }
    override fun bltzl(i: Int, s: InstructionInfo) = s.branchLikely { RS lt 0.lit }
    override fun blezl(i: Int, s: InstructionInfo) = s.branchLikely { RS le 0.lit }
    override fun bgtzl(i: Int, s: InstructionInfo) = s.branchLikely { RS gt 0.lit }
    override fun bgezl(i: Int, s: InstructionInfo) = s.branchLikely { RS ge 0.lit }
    //override fun bgezall(i: Int, s: InstructionInfo) = s.branchLikely { RA = _nPC + 4; RS ge 0.lit }
    //override fun bltzall(i: Int, s: InstructionInfo) = s.branchLikely { RA = _nPC + 4; RS lt 0.lit }

    override fun bc1f(i: Int, s: InstructionInfo) = s.branch { CpuState::f_get_fcr31_cc_not.invoke(p0) }
    override fun bc1t(i: Int, s: InstructionInfo) = s.branch { CpuState::f_get_fcr31_cc.invoke(p0) }
    override fun bc1fl(i: Int, s: InstructionInfo) = s.branchLikely { CpuState::f_get_fcr31_cc_not.invoke(p0) }
    override fun bc1tl(i: Int, s: InstructionInfo) = s.branchLikely { CpuState::f_get_fcr31_cc.invoke(p0) }

    //override fun bc1f(i: Int, s: InstructionInfo) = s.branch { fcr31_cc.not() }
    //override fun bc1t(i: Int, s: InstructionInfo) = s.branch { fcr31_cc }
    //override fun bc1fl(i: Int, s: InstructionInfo) = s.branchLikely { fcr31_cc.not() }
    //override fun bc1tl(i: Int, s: InstructionInfo) = s.branchLikely { fcr31_cc }

    override fun j(i: Int, s: InstructionInfo) = s.jump { real_jump_address.lit } // 0xf0000000
    override fun jr(i: Int, s: InstructionInfo) = s.jump { RS }
    override fun jal(i: Int, s: InstructionInfo) = s.jump({ RA = (ii.PC + 8).lit }) { real_jump_address.lit }
    override fun jalr(i: Int, s: InstructionInfo) = s.jump({ RD = (ii.PC + 8).lit }) { RS }

    // Float
    override fun cfc1(i: Int, s: InstructionInfo) = s { RT = CpuState::cfc1.invoke(p0, IR.rd.lit, RT) }
    override fun ctc1(i: Int, s: InstructionInfo) = s { STM(CpuState::ctc1.invoke(p0, IR.rd.lit, RT)) }

    override fun mfc1(i: Int, s: InstructionInfo) = s { RT = FS_I }
    override fun mtc1(i: Int, s: InstructionInfo) = s { FS_I = RT }

    override fun cvt_s_w(i: Int, s: InstructionInfo) = s { FD = CpuState::cvt_s_w.invoke(p0, FS_I) }
    override fun cvt_w_s(i: Int, s: InstructionInfo) = s { FD_I = CpuState::cvt_w_s.invoke(p0, FS) }

    override fun trunc_w_s(i: Int, s: InstructionInfo) = s { FD_I = CpuState::trunc_w_s.invoke(p0, FS) }
    override fun round_w_s(i: Int, s: InstructionInfo) = s { FD_I = CpuState::round_w_s.invoke(p0, FS) }
    override fun ceil_w_s(i: Int, s: InstructionInfo) = s { FD_I = CpuState::ceil_w_s.invoke(p0, FS) }
    override fun floor_w_s(i: Int, s: InstructionInfo) = s { FD_I = CpuState::floor_w_s.invoke(p0, FS) }

    override fun mov_s(i: Int, s: InstructionInfo) = s.checkNan { FD = CpuState::fmov.invoke(p0, FS) }
    override fun add_s(i: Int, s: InstructionInfo) = s.checkNan { FD = CpuState::fadd.invoke(p0, FS, FT) }
    override fun sub_s(i: Int, s: InstructionInfo) = s.checkNan { FD = CpuState::fsub.invoke(p0, FS, FT) }
    override fun mul_s(i: Int, s: InstructionInfo) = s.checkNan { FD = CpuState::fmul.invoke(p0, FS, FT) }
    override fun div_s(i: Int, s: InstructionInfo) = s.checkNan { FD = CpuState::fdiv.invoke(p0, FS, FT) }
    override fun neg_s(i: Int, s: InstructionInfo) = s.checkNan { FD = CpuState::fneg.invoke(p0, FS) }
    override fun abs_s(i: Int, s: InstructionInfo) = s.checkNan { FD = CpuState::fabs.invoke(p0, FS) }
    override fun sqrt_s(i: Int, s: InstructionInfo) = s.checkNan { FD = CpuState::fsqrt.invoke(p0, FS) }

    override fun c_f_s(i: Int, s: InstructionInfo)    = s { fcr31_cc = CpuState::c_f_s.invoke(p0, FS, FT) }
    override fun c_un_s(i: Int, s: InstructionInfo)   = s { fcr31_cc = CpuState::c_un_s.invoke(p0, FS, FT) }
    override fun c_eq_s(i: Int, s: InstructionInfo)   = s { fcr31_cc = CpuState::c_eq_s.invoke(p0, FS, FT) }
    override fun c_ueq_s(i: Int, s: InstructionInfo)  = s { fcr31_cc = CpuState::c_ueq_s.invoke(p0, FS, FT) }
    override fun c_olt_s(i: Int, s: InstructionInfo)  = s { fcr31_cc = CpuState::c_olt_s.invoke(p0, FS, FT) }
    override fun c_ult_s(i: Int, s: InstructionInfo)  = s { fcr31_cc = CpuState::c_ult_s.invoke(p0, FS, FT) }
    override fun c_ole_s(i: Int, s: InstructionInfo)  = s { fcr31_cc = CpuState::c_ole_s.invoke(p0, FS, FT) }
    override fun c_ule_s(i: Int, s: InstructionInfo)  = s { fcr31_cc = CpuState::c_ule_s.invoke(p0, FS, FT) }

    override fun c_sf_s(i: Int, s: InstructionInfo)   = s { fcr31_cc = CpuState::c_f_s.invoke(p0, FS, FT) }
    override fun c_ngle_s(i: Int, s: InstructionInfo) = s { fcr31_cc = CpuState::c_un_s.invoke(p0, FS, FT) }
    override fun c_seq_s(i: Int, s: InstructionInfo)  = s { fcr31_cc = CpuState::c_eq_s.invoke(p0, FS, FT) }
    override fun c_ngl_s(i: Int, s: InstructionInfo)  = s { fcr31_cc = CpuState::c_ueq_s.invoke(p0, FS, FT) }
    override fun c_lt_s(i: Int, s: InstructionInfo)   = s { fcr31_cc = CpuState::c_olt_s.invoke(p0, FS, FT) }
    override fun c_nge_s(i: Int, s: InstructionInfo)  = s { fcr31_cc = CpuState::c_ult_s.invoke(p0, FS, FT) }
    override fun c_le_s(i: Int, s: InstructionInfo)   = s { fcr31_cc = CpuState::c_ole_s.invoke(p0, FS, FT) }
    override fun c_ngt_s(i: Int, s: InstructionInfo)  = s { fcr31_cc = CpuState::c_ule_s.invoke(p0, FS, FT) }
}

data class InstructionInfo(var PC: Int, var IR: Int)

typealias CpuStateStmBuilder = StmBuilder<Unit, CpuState, Unit>
typealias CpuStateFunction = DFunction1<Unit, CpuState>

open class BaseDynarecMethodBuilder : InstructionEvaluator<InstructionInfo>() {
    var sstms = StmBuilder(Unit::class, CpuState::class, Unit::class)
    val dispatcher = InstructionDispatcher(this)

    fun generateFunction() = DFunction1(DVOID, DClass(CpuState::class), sstms.build())

    @PublishedApi
    internal val ii = InstructionInfo(0, 0)
    val IR get() = ii.IR
    //val PC get() = ii.PC

    inline fun CpuStateStmBuilder.useTemp(callback: CpuStateStmBuilder.() -> Unit): CpuStateStmBuilder {
        val old = this@BaseDynarecMethodBuilder.sstms
        this@BaseDynarecMethodBuilder.sstms = this
        try {
            callback(this)
        } finally {
            this@BaseDynarecMethodBuilder.sstms = old
        }
        return this
    }

    var reachedFlow = false

    fun dispatch(pc: Int, i: Int) {
        ii.PC = pc
        ii.IR = i

        val delayed = this.delayed
        if (delayed != null) {
            val oldStms = sstms
            try {
                sstms = StmBuilder(Unit::class, CpuState::class, Unit::class)
                dispatcher.dispatch(ii, pc, i)
                val newStms = sstms
                sstms = oldStms

                sstms.run {
                    val cond = delayed.cond
                    if (cond != null) {
                        IF(cond) {
                            STM(newStms.build())
                            useTemp {
                                PC = delayed.jumpTo
                            }
                            RET()
                        } ELSE {
                            useTemp {
                                PC = (ii.PC + 4).lit
                            }
                            if (!delayed.likely) {
                                STM(newStms.build())
                            }
                            RET()
                        }
                    } else {
                        if (reachedFlow) {
                            useTemp {
                                PC = delayed.jumpTo
                                delayed.after?.let { STM(it) }
                            }
                            STM(newStms.build())
                            RET()
                        } else {
                            useTemp {
                                delayed.after?.let { STM(it) }
                            }
                            STM(newStms.build())
                            useTemp {
                                PC = delayed.jumpTo
                            }
                            RET()
                        }
                    }
                }
            } finally {
                sstms = oldStms
                this.delayed = null
                reachedFlow = true
            }
        } else {
            dispatcher.dispatch(ii, pc, i)
        }
    }

    fun CpuStateStmBuilder.getRegister(n: Int): DExpr<Int> {
        return when (n) {
            0 -> 0.lit
            else -> p0[CpuState.getGprProp(n)]
        }
    }

    fun CpuStateStmBuilder.setRegister(n: Int, value: DExpr<Int>) {
        if (n != 0) SET(p0[CpuState.getGprProp(n)], value)
    }

    fun CpuStateStmBuilder.setRegisterFI(n: Int, value: DExpr<Int>) {
        STM(CpuState::setFprI.invoke(p0, n.lit, value))
    }

    fun CpuStateStmBuilder.getRegisterFI(n: Int): DExpr<Int> {
        return CpuState::getFprI.invoke(p0, n.lit)
    }

    fun CpuStateStmBuilder.setRegisterF(n: Int, value: DExpr<Float>) {
        STM(CpuState::setFpr.invoke(p0, n.lit, value))
    }

    fun CpuStateStmBuilder.getRegisterF(n: Int): DExpr<Float> {
        return CpuState::getFpr.invoke(p0, n.lit)
    }

    ///////////////////////////////////
    var RD: DExpr<Int>
        set(value) = sstms.run { setRegister(IR.rd, value) }
        get() = sstms.run { getRegister(IR.rd) }

    var RS: DExpr<Int>
        set(value) = sstms.run { setRegister(IR.rs, value) }
        get() = sstms.run { getRegister(IR.rs) }

    var RT: DExpr<Int>
        set(value) = sstms.run { setRegister(IR.rt, value) }
        get() = sstms.run { getRegister(IR.rt) }

    ///////////////////////////////////
    var FS_I: DExpr<Int>
        set(value) = sstms.run { setRegisterFI(IR.fs, value) }
        get() = sstms.run { getRegisterFI(IR.fs) }

    var FD_I: DExpr<Int>
        set(value) = sstms.run { setRegisterFI(IR.fd, value) }
        get() = sstms.run { getRegisterFI(IR.fd) }

    var FT_I: DExpr<Int>
        set(value) = sstms.run { setRegisterFI(IR.ft, value) }
        get() = sstms.run { getRegisterFI(IR.ft) }

    ///////////////////////////////////
    var FS: DExpr<Float>
        set(value) = sstms.run { setRegisterF(IR.fs, value) }
        get() = sstms.run { getRegisterF(IR.fs) }

    var FD: DExpr<Float>
        set(value) = sstms.run { setRegisterF(IR.fd, value) }
        get() = sstms.run { getRegisterF(IR.fd) }

    var FT: DExpr<Float>
        set(value) = sstms.run { setRegisterF(IR.ft, value) }
        get() = sstms.run { getRegisterF(IR.ft) }

    var fcr31_cc: DExpr<Boolean>
        set(value) = sstms.run { p0[CpuState::fcr31_cc] = value }
        get() = sstms.run { p0[CpuState::fcr31_cc] }


    ///////////////////////////////////
    var LO: DExpr<Int>
        set(value) = sstms.run { SET(p0[CpuState::LO], value) }
        get() = sstms.run { p0[CpuState::LO] }

    var HI: DExpr<Int>
        set(value) = sstms.run { SET(p0[CpuState::HI], value) }
        get() = sstms.run { p0[CpuState::HI] }

    ///////////////////////////////////
    var IC: DExpr<Int>
        set(value) = sstms.run { SET(p0[CpuState::IC], value) }
        get() = sstms.run { p0[CpuState::IC] }

    var PC: DExpr<Int>
        set(value) = sstms.run {
            STM(CpuState::setPC.invoke(p0, value))
        }
        get() = sstms.run { p0[CpuState::_PC] }

    var RA: DExpr<Int>
        set(value) = sstms.run { SET(p0[CpuState::r31], value) }
        get() = sstms.run { p0[CpuState::r31] }

    val POS: DExpr<Int> get() = sstms.run { IR.pos.lit }
    val SIZE_E: DExpr<Int> get() = sstms.run { IR.size_e.lit }
    val SIZE_I: DExpr<Int> get() = sstms.run { IR.size_i.lit }
    val S_IMM16: DExpr<Int> get() = sstms.run { IR.s_imm16.lit }
    val U_IMM16: DExpr<Int> get() = sstms.run { IR.u_imm16.lit }
    val SYSCALL: DExpr<Int> get() = sstms.run { IR.syscall.lit }
    val JUMP_ADDRESS: DExpr<Int> get() = sstms.run { IR.jump_address.lit }

    val RS_IMM16 get() = sstms.run { RS + S_IMM16 }
    val real_jump_address get() = (ii.PC and (-268435456)) or IR.jump_address


    val S_IMM16_V: Int get() = sstms.run { IR.s_imm16 }
    val U_IMM16_V: Int get() = sstms.run { IR.u_imm16 }

    //fun set_pc(value: Int) {
    //    sstms.run {
    //        SET(p0[CpuState::_PC], value.lit)
    //        SET(p0[CpuState::_nPC], value.lit)
    //    }
    //}

    inline operator fun InstructionInfo.invoke(callback: CpuStateStmBuilder.(InstructionInfo) -> Unit): Unit {
        callback(sstms, this)
    }

    inline fun InstructionInfo.checkNan(callback: CpuStateStmBuilder.(InstructionInfo) -> Unit): Unit {
        callback(sstms, this)
        sstms.run {
            STM(CpuState::_checkFNan.invoke(p0, FD))
        }
    }

    inline fun InstructionInfo.preadvanceAndFlow(callback: CpuStateStmBuilder.(InstructionInfo) -> Unit): Unit {
        if (delayed == null) {
            this@BaseDynarecMethodBuilder.PC = sstms.run { (ii.PC + 4).lit }
        }
        callback(sstms, this)
        reachedFlow = true
    }

    data class Delayed(val cond: DExpr<Boolean>?, val jumpTo: DExpr<Int>, val after: DStm?, val likely: Boolean)

    @PublishedApi
    internal var delayed: Delayed? = null

    inline fun InstructionInfo.branch(callback: CpuStateStmBuilder.(InstructionInfo) -> DExpr<Boolean>): Unit =
        _branch(likely = false, callback = callback)

    inline fun InstructionInfo.branchLikely(callback: CpuStateStmBuilder.(InstructionInfo) -> DExpr<Boolean>): Unit =
        _branch(likely = true, callback = callback)

    inline fun InstructionInfo._branch(
        likely: Boolean = false,
        callback: CpuStateStmBuilder.(InstructionInfo) -> DExpr<Boolean>
    ): Unit {
        sstms.run {
            delayed = Delayed(callback(this, this@_branch), (ii.PC + IR.s_imm16 * 4 + 4).lit, null, likely = likely)
        }
    }

    inline fun InstructionInfo.jump(
        noinline exec: (CpuStateStmBuilder.() -> Unit)? = null,
        callback: CpuStateStmBuilder.(InstructionInfo) -> DExpr<Int>
    ): Unit {
        sstms.run {
            val extraStm = if (exec != null) {
                StmBuilder(Unit::class, CpuState::class, Unit::class).useTemp {
                    exec()
                }.build()
            } else {
                null
            }
            delayed = Delayed(null, callback(this, this@jump), extraStm, likely = false)
        }
    }
}
