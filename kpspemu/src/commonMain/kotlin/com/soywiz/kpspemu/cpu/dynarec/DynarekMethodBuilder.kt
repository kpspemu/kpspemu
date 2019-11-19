package com.soywiz.kpspemu.cpu.dynarec

import com.soywiz.dynarek2.*
import com.soywiz.kpspemu.cpu.*

class DynarekMethodBuilder : BaseDynarecMethodBuilder() {

    // ALU
    override fun lui(i: Int, s: InstructionInfo) = s { RT = ((IR.u_imm16 shl 16).lit) }

    override fun movz(i: Int, s: InstructionInfo) = s {
        //RD = ::dyna_movz.invoke(RT, RD, RS)
        IF (RT EQ 0.lit) { RD = RS }
    }
    override fun movn(i: Int, s: InstructionInfo) = s {
        //RD = ::dyna_movn.invoke(RT, RD, RS)
        IF (RT NE 0.lit) { RD = RS }
    }

    override fun ext(i: Int, s: InstructionInfo) = s { RT = ::dyna_ext.invoke(RS, POS, SIZE_E) }
    override fun ins(i: Int, s: InstructionInfo) = s { RT = ::dyna_ins.invoke(RT, RS, POS, SIZE_I) }

    override fun clz(i: Int, s: InstructionInfo) = s { RD = ::dyna_clz.invoke(RS) }
    override fun clo(i: Int, s: InstructionInfo) = s { RD = ::dyna_clo.invoke(RS) }
    override fun seb(i: Int, s: InstructionInfo) = s { RD = ::dyna_seb.invoke(RT) }
    override fun seh(i: Int, s: InstructionInfo) = s { RD = ::dyna_seh.invoke(RT) }

    override fun wsbh(i: Int, s: InstructionInfo) = s { RD = ::dyna_wsbh.invoke(RT) }
    override fun wsbw(i: Int, s: InstructionInfo) = s { RD = ::dyna_wsbw.invoke(RT) }

    override fun max(i: Int, s: InstructionInfo) = s { RD = ::dyna_max.invoke(RS, RT) }
    override fun min(i: Int, s: InstructionInfo) = s { RD = ::dyna_min.invoke(RS, RT) }

    override fun add(i: Int, s: InstructionInfo) = s { RD = RS + RT }
    override fun addu(i: Int, s: InstructionInfo) = s { RD = RS + RT }
    override fun sub(i: Int, s: InstructionInfo) = s { RD = RS - RT }
    override fun subu(i: Int, s: InstructionInfo) = s { RD = RS - RT }
    override fun addi(i: Int, s: InstructionInfo) = s { RT = RS + S_IMM16 }
    override fun addiu(i: Int, s: InstructionInfo) = s { RT = RS + S_IMM16 }

    override fun div(i: Int, s: InstructionInfo) = s {
        LO = RS / RT
        HI = RS % RT
    }

    override fun divu(i: Int, s: InstructionInfo) = s {
        LO = ::dyna_divu_LO.invoke(RS, RT)
        HI = ::dyna_divu_HI.invoke(RS, RT)
    }

    override fun mult(i: Int, s: InstructionInfo) = s {
        //HI_LO = ::dyna_mult.invoke(RS, RT)
        LO = ::dyna_mult_LO.invoke(RS, RT)
        HI = ::dyna_mult_HI.invoke(RS, RT)
    }

    override fun multu(i: Int, s: InstructionInfo) = s {
        LO = ::dyna_multu_LO.invoke(RS, RT)
        HI = ::dyna_multu_HI.invoke(RS, RT)
    }

    override fun madd(i: Int, s: InstructionInfo) = s { STM(::dyna_madd.invoke(EXTERNAL, RS, RT)) }
    override fun maddu(i: Int, s: InstructionInfo) = s { STM(::dyna_maddu.invoke(EXTERNAL, RS, RT)) }
    override fun msub(i: Int, s: InstructionInfo) = s { STM(::dyna_msub.invoke(EXTERNAL, RS, RT)) }
    override fun msubu(i: Int, s: InstructionInfo) = s { STM(::dyna_msubu.invoke(EXTERNAL, RS, RT)) }

    override fun mflo(i: Int, s: InstructionInfo) = s { RD = LO }
    override fun mfhi(i: Int, s: InstructionInfo) = s { RD = HI }
    override fun mfic(i: Int, s: InstructionInfo) = s { RT = IC }

    override fun mtlo(i: Int, s: InstructionInfo) = s { LO = RS }
    override fun mthi(i: Int, s: InstructionInfo) = s { HI = RS }
    override fun mtic(i: Int, s: InstructionInfo) = s { IC = RT }

    // ALU: Bit
    override fun or(i: Int, s: InstructionInfo) = s { RD = RS OR RT }

    override fun xor(i: Int, s: InstructionInfo) = s { RD = RS XOR RT }
    override fun and(i: Int, s: InstructionInfo) = s { RD = RS AND RT }
    override fun nor(i: Int, s: InstructionInfo) = s { RD = INV(RS OR RT) }
    override fun ori(i: Int, s: InstructionInfo) = s { RT = RS OR U_IMM16 }
    override fun xori(i: Int, s: InstructionInfo) = s { RT = RS XOR U_IMM16 }
    override fun andi(i: Int, s: InstructionInfo) = s { RT = RS AND U_IMM16 }

    override fun sll(i: Int, s: InstructionInfo) = s { RD = ::dyna_sll.invoke(RT, POS) }
    override fun sra(i: Int, s: InstructionInfo) = s { RD = ::dyna_sra.invoke(RT, POS) }
    override fun srl(i: Int, s: InstructionInfo) = s { RD = ::dyna_srl.invoke(RT, POS) }
    override fun sllv(i: Int, s: InstructionInfo) = s { RD = ::dyna_sll.invoke(RT, RS) }
    override fun srav(i: Int, s: InstructionInfo) = s { RD = ::dyna_sra.invoke(RT, RS) }
    override fun srlv(i: Int, s: InstructionInfo) = s { RD = ::dyna_srl.invoke(RT, RS) }
    override fun bitrev(i: Int, s: InstructionInfo) = s { RD = ::dyna_bitrev32.invoke(RT) }
    override fun rotr(i: Int, s: InstructionInfo) = s { RD = ::dyna_rotr.invoke(RT, POS) }
    override fun rotrv(i: Int, s: InstructionInfo) = s { RD = ::dyna_rotr.invoke(RT, RS) }

    fun D2Builder.raddress(address: D2ExprI, shift: Int): D2ExprI {
        val raddr = address AND 0x0FFFFFFF.lit
        return if (shift != 0) (raddr USHR shift.lit) else raddr
    }

    fun D2Builder.MEM8(address: D2ExprI) = D2Expr.RefI(D2MemSlot.MEM, D2Size.BYTE, raddress(address, 0))
    fun D2Builder.U_MEM8(address: D2ExprI) = MEM8(address) AND 0xFF.lit

    fun D2Builder.MEM16(address: D2ExprI) = D2Expr.RefI(D2MemSlot.MEM, D2Size.SHORT, raddress(address, 1))
    fun D2Builder.U_MEM16(address: D2ExprI) = MEM16(address) AND 0xFFFF.lit

    fun D2Builder.MEM32(address: D2ExprI) = D2Expr.RefI(D2MemSlot.MEM, D2Size.INT, raddress(address, 2))

    // Memory
    override fun lb(i: Int, s: InstructionInfo) = s { RT = MEM8(RS_IMM16) }

    override fun lbu(i: Int, s: InstructionInfo) = s { RT = U_MEM8(RS_IMM16) }
    override fun lh(i: Int, s: InstructionInfo) = s { RT = MEM16(RS_IMM16) }
    override fun lhu(i: Int, s: InstructionInfo) = s { RT = U_MEM16(RS_IMM16) }
    override fun lw(i: Int, s: InstructionInfo) = s { RT = MEM32(RS_IMM16) }
    override fun ll(i: Int, s: InstructionInfo) = s { RT = MEM32(RS_IMM16) }

    //override fun lwl(i: Int, s: InstructionInfo) = s { RT = CpuState::lwl.invoke(p0, RS_IMM16, RT) }
    //override fun lwr(i: Int, s: InstructionInfo) = s { RT = CpuState::lwr.invoke(p0, RS_IMM16, RT) }
    //
    //override fun swl(i: Int, s: InstructionInfo) = s { STM(CpuState::swl.invoke(p0, RS_IMM16, RT)) }
    //override fun swr(i: Int, s: InstructionInfo) = s { STM(CpuState::swr.invoke(p0, RS_IMM16, RT)) }

    override fun sb(i: Int, s: InstructionInfo) = s { SET(MEM8(RS_IMM16), RT) }
    override fun sh(i: Int, s: InstructionInfo) = s { SET(MEM16(RS_IMM16), RT) }
    override fun sw(i: Int, s: InstructionInfo) = s { SET(MEM32(RS_IMM16), RT) }
    override fun sc(i: Int, s: InstructionInfo) = s { SET(MEM32(RS_IMM16), RT); RT = 1.lit }

    override fun lwc1(i: Int, s: InstructionInfo) = s { FT_I = MEM32(RS_IMM16) }
    override fun swc1(i: Int, s: InstructionInfo) = s { SET(MEM32(RS_IMM16), FT_I) }

    // Special
    override fun syscall(i: Int, s: InstructionInfo) =
        s.preadvanceAndFlow {
            TEMP0 = ::dyna_syscall.invoke(EXTERNAL, SYSCALL)
            IF(TEMP0 NE 0.lit) {
                RETURN(TEMP0)
            }
        }

    override fun _break(i: Int, s: InstructionInfo) = s.preadvanceAndFlow {
        //STM(::dyna_break.invoke(SYSCALL))
        RETURN(SYSCALL)
    }

    // Set less
    override fun slt(i: Int, s: InstructionInfo) = s { RD = ::dyna_slt.invoke(RS, RT) }

    override fun sltu(i: Int, s: InstructionInfo) = s { RD = ::dyna_sltu.invoke(RS, RT) }
    override fun slti(i: Int, s: InstructionInfo) = s { RT = ::dyna_slt.invoke(RS, S_IMM16) }
    override fun sltiu(i: Int, s: InstructionInfo) = s { RT = ::dyna_sltu.invoke(RS, S_IMM16) }

    // Branch
    override fun beq(i: Int, s: InstructionInfo) = s.branch { RS EQ RT }

    override fun bne(i: Int, s: InstructionInfo) = s.branch { RS NE RT }
    override fun bltz(i: Int, s: InstructionInfo) = s.branch { RS LT 0.lit }
    override fun blez(i: Int, s: InstructionInfo) = s.branch { RS LE 0.lit }
    override fun bgtz(i: Int, s: InstructionInfo) = s.branch { RS GT 0.lit }
    override fun bgez(i: Int, s: InstructionInfo) = s.branch { RS GE 0.lit }

    override fun beql(i: Int, s: InstructionInfo) = s.branchLikely { RS EQ RT }
    override fun bnel(i: Int, s: InstructionInfo) = s.branchLikely { RS NE RT }
    override fun bltzl(i: Int, s: InstructionInfo) = s.branchLikely { RS LT 0.lit }
    override fun blezl(i: Int, s: InstructionInfo) = s.branchLikely { RS LE 0.lit }
    override fun bgtzl(i: Int, s: InstructionInfo) = s.branchLikely { RS GT 0.lit }
    override fun bgezl(i: Int, s: InstructionInfo) = s.branchLikely { RS GE 0.lit }

    // @TODO: This is probably wrong!!
    override fun bgezal(i: Int, s: InstructionInfo) = s.branch { RA = _nPC + 4.lit; RS GE 0.lit }
    override fun bltzal(i: Int, s: InstructionInfo) = s.branch { RA = _nPC + 4.lit; RS LT 0.lit }
    override fun bgezall(i: Int, s: InstructionInfo) = s.branchLikely { RA = _nPC + 4.lit; RS GE 0.lit }
    override fun bltzall(i: Int, s: InstructionInfo) = s.branchLikely { RA = _nPC + 4.lit; RS LT 0.lit }

    //override fun bc1f(i: Int, s: InstructionInfo) = s.branch { CpuState::f_get_fcr31_cc_not.invoke(p0) }
    //override fun bc1t(i: Int, s: InstructionInfo) = s.branch { CpuState::f_get_fcr31_cc.invoke(p0) }
    //override fun bc1fl(i: Int, s: InstructionInfo) = s.branchLikely { CpuState::f_get_fcr31_cc_not.invoke(p0) }
    //override fun bc1tl(i: Int, s: InstructionInfo) = s.branchLikely { CpuState::f_get_fcr31_cc.invoke(p0) }

    //override fun bc1f(i: Int, s: InstructionInfo) = s.branch { fcr31_cc.not() }
    //override fun bc1t(i: Int, s: InstructionInfo) = s.branch { fcr31_cc }
    //override fun bc1fl(i: Int, s: InstructionInfo) = s.branchLikely { fcr31_cc.not() }
    //override fun bc1tl(i: Int, s: InstructionInfo) = s.branchLikely { fcr31_cc }

    override fun j(i: Int, s: InstructionInfo) = s.jump { real_jump_address.lit } // 0xf0000000
    override fun jr(i: Int, s: InstructionInfo) = s.jump { RS }
    override fun jal(i: Int, s: InstructionInfo) = s.jump({ RA = (ii.PC + 8).lit }) { real_jump_address.lit }
    override fun jalr(i: Int, s: InstructionInfo) = s.jump({ RD = (ii.PC + 8).lit }) { RS }

    // Float
    //override fun cfc1(i: Int, s: InstructionInfo) = s { RT = CpuState::cfc1.invoke(p0, IR.rd.lit, RT) }
    //override fun ctc1(i: Int, s: InstructionInfo) = s { STM(CpuState::ctc1.invoke(p0, IR.rd.lit, RT)) }

    override fun mfc1(i: Int, s: InstructionInfo) = s { RT = FS_I }
    override fun mtc1(i: Int, s: InstructionInfo) = s { FS_I = RT }

    override fun cvt_s_w(i: Int, s: InstructionInfo) = s { FD = ::dyna_cvt_s_w.invoke(FS_I) }
    override fun cvt_w_s(i: Int, s: InstructionInfo) = s { FD_I = ::dyna_cvt_w_s.invoke(EXTERNAL, FS) }

    override fun trunc_w_s(i: Int, s: InstructionInfo) = s { FD_I = ::dyna_trunc_w_s.invoke(FS) }
    override fun round_w_s(i: Int, s: InstructionInfo) = s { FD_I = ::dyna_round_w_s.invoke(FS) }
    override fun ceil_w_s(i: Int, s: InstructionInfo) = s { FD_I = ::dyna_ceil_w_s.invoke(FS) }
    override fun floor_w_s(i: Int, s: InstructionInfo) = s { FD_I = ::dyna_floor_w_s.invoke(FS) }

    override fun mov_s(i: Int, s: InstructionInfo) = s.checkNan { FD = FS }
    override fun add_s(i: Int, s: InstructionInfo) = s.checkNan { FD = FS + FT }
    override fun sub_s(i: Int, s: InstructionInfo) = s.checkNan { FD = FS - FT }
    override fun mul_s(i: Int, s: InstructionInfo) = s.checkNan { FD = ::dyna_fmul.invoke(EXTERNAL, FS, FT) }
    override fun div_s(i: Int, s: InstructionInfo) = s.checkNan { FD = FS / FT }
    override fun neg_s(i: Int, s: InstructionInfo) = s.checkNan { FD = ::dyna_fneg.invoke(FS) }
    override fun abs_s(i: Int, s: InstructionInfo) = s.checkNan { FD = ::dyna_fabs.invoke(FS) }
    override fun sqrt_s(i: Int, s: InstructionInfo) = s.checkNan { FD = ::dyna_fsqrt.invoke(FS) }

    override fun c_f_s(i: Int, s: InstructionInfo) = s { fcr31_cc = ::dyna_c_f_s.invoke(FS, FT) }
    override fun c_un_s(i: Int, s: InstructionInfo) = s { fcr31_cc = ::dyna_c_un_s.invoke(FS, FT) }
    override fun c_eq_s(i: Int, s: InstructionInfo) = s { fcr31_cc = ::dyna_c_eq_s.invoke(FS, FT) }
    override fun c_ueq_s(i: Int, s: InstructionInfo) = s { fcr31_cc = ::dyna_c_ueq_s.invoke(FS, FT) }
    override fun c_olt_s(i: Int, s: InstructionInfo) = s { fcr31_cc = ::dyna_c_olt_s.invoke(FS, FT) }
    override fun c_ult_s(i: Int, s: InstructionInfo) = s { fcr31_cc = ::dyna_c_ult_s.invoke(FS, FT) }
    override fun c_ole_s(i: Int, s: InstructionInfo) = s { fcr31_cc = ::dyna_c_ole_s.invoke(FS, FT) }
    override fun c_ule_s(i: Int, s: InstructionInfo) = s { fcr31_cc = ::dyna_c_ule_s.invoke(FS, FT) }

    override fun c_sf_s(i: Int, s: InstructionInfo) = s { fcr31_cc = ::dyna_c_f_s.invoke(FS, FT) }
    override fun c_ngle_s(i: Int, s: InstructionInfo) = s { fcr31_cc = ::dyna_c_un_s.invoke(FS, FT) }
    override fun c_seq_s(i: Int, s: InstructionInfo) = s { fcr31_cc = ::dyna_c_eq_s.invoke(FS, FT) }
    override fun c_ngl_s(i: Int, s: InstructionInfo) = s { fcr31_cc = ::dyna_c_ueq_s.invoke(FS, FT) }
    override fun c_lt_s(i: Int, s: InstructionInfo) = s { fcr31_cc = ::dyna_c_olt_s.invoke(FS, FT) }
    override fun c_nge_s(i: Int, s: InstructionInfo) = s { fcr31_cc = ::dyna_c_ult_s.invoke(FS, FT) }
    override fun c_le_s(i: Int, s: InstructionInfo) = s { fcr31_cc = ::dyna_c_ole_s.invoke(FS, FT) }
    override fun c_ngt_s(i: Int, s: InstructionInfo) = s { fcr31_cc = ::dyna_c_ule_s.invoke(FS, FT) }
}

data class InstructionInfo(var PC: Int, var IR: Int)

open class BaseDynarecMethodBuilder : InstructionEvaluator<InstructionInfo>() {
    //var sstms = StmBuilder(Unit::class, CpuState::class, Unit::class)
    var sstms = D2Builder()
    val dispatcher = InstructionDispatcher(this)

    //fun generateFunction(): DFunction1 = DFunction1(DVOID, DClass(CpuState::class), sstms.build())
    fun generateFunction(): D2Func = D2Func(sstms.build())

    @PublishedApi
    internal val ii = InstructionInfo(0, 0)
    val IR get() = ii.IR
    //val PC get() = ii.PC

    inline fun D2Builder.useTemp(callback: D2Builder.() -> Unit): D2Builder {
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
                sstms = D2Builder()
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
                            RET_VOID()
                        } ELSE {
                            useTemp {
                                incrementPC()
                            }
                            if (!delayed.likely) {
                                STM(newStms.build())
                            }
                            RET_VOID()
                        }
                    } else {
                        if (reachedFlow) {
                            useTemp {
                                PC = delayed.jumpTo
                                delayed.after?.let { STM(it) }
                            }
                            STM(newStms.build())
                            RET_VOID()
                        } else {
                            useTemp {
                                delayed.after?.let { STM(it) }
                            }
                            STM(newStms.build())
                            useTemp {
                                PC = delayed.jumpTo
                            }
                            RET_VOID()
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

    fun reg(index: Int): D2Expr.Ref<Int> = sstms { REGS32(index) }
    fun setReg(index: Int, value: D2ExprI) = sstms { SET(reg(index), value) }

    fun reg64(index: Int): D2Expr.Ref<Long> = sstms { REGS64(index) }
    fun setReg64(index: Int, value: D2ExprL) = sstms { SET(reg64(index), value) }

    fun gpr(index: Int): D2Expr.Ref<Int> = reg(RegOff.GPR(index))
    fun setGpr(index: Int, value: D2ExprI) = sstms { SET(gpr(index), value) }

    fun fprI(index: Int): D2Expr.Ref<Int> = reg(RegOff.FPR(index))
    fun setFprI(index: Int, value: D2ExprI) = sstms { SET(fprI(index), value) }

    fun fpr(index: Int): D2Expr.Ref<Float> = sstms { REGF32(RegOff.FPR(index)) }
    fun setFpr(index: Int, value: D2ExprF) = sstms { SET(fpr(index), value) }

    ///////////////////////////////////

    var TEMP0: D2ExprI; set(value) = setReg(RegOff.TEMP0, value); get() = reg(RegOff.TEMP0)

    var RD: D2ExprI; set(value) = setGpr(IR.rd, value); get() = gpr(IR.rd)
    var RS: D2ExprI; set(value) = setGpr(IR.rs, value); get() = gpr(IR.rs)
    var RT: D2ExprI; set(value) = setGpr(IR.rt, value); get() = gpr(IR.rt)

    var FS_I: D2ExprI; set(value) = setFprI(IR.fs, value); get() = fprI(IR.fs)
    var FD_I: D2ExprI; set(value) = setFprI(IR.fd, value); get() = fprI(IR.fd)
    var FT_I: D2ExprI; set(value) = setFprI(IR.ft, value); get() = fprI(IR.ft)

    var FS: D2ExprF; set(value) = setFpr(IR.fs, value); get() = fpr(IR.fs)
    var FD: D2ExprF; set(value) = setFpr(IR.fd, value); get() = fpr(IR.fd)
    var FT: D2ExprF; set(value) = setFpr(IR.ft, value); get() = fpr(IR.ft)

    var fcr31: D2ExprI; set(value) = setReg(RegOff.FCR_CC(31), value); get() = reg(RegOff.FCR_CC(31))

    var fcr31_cc: D2ExprB
        set(value) = sstms { fcr31 = fcr31.INSERT(23, 1, value) }
        get() = sstms { fcr31.EXTRACT(23, 1) }


    ///////////////////////////////////
    var LO: D2ExprI set(value) = setReg(RegOff.LO, value); get() = reg(RegOff.LO)
    var HI: D2ExprI set(value) = setReg(RegOff.HI, value); get() = reg(RegOff.HI)
    var IC: D2ExprI set(value) = setReg(RegOff.IC, value); get() = reg(RegOff.IC)

    var HI_LO: D2ExprL set(value) = setReg64(RegOff.HI_LO, value); get() = reg64(RegOff.HI_LO)

    var _PC: D2ExprI set(value) = setReg(RegOff.PC, value); get() = reg(RegOff.PC)
    var _nPC: D2ExprI set(value) = setReg(RegOff.nPC, value); get() = reg(RegOff.nPC)

    var PC: D2ExprI
        set(value) = sstms { _PC = value; _nPC = value + 4.lit }
        get() = sstms { _PC }

    var RA: D2ExprI set(value) = setReg(RegOff.RA, value); get() = reg(RegOff.RA)

    val POS get() = sstms { IR.pos.lit }
    val SIZE_E get() = sstms { IR.size_e.lit }
    val SIZE_I get() = sstms { IR.size_i.lit }
    val S_IMM16 get() = sstms { IR.s_imm16.lit }
    val U_IMM16 get() = sstms { IR.u_imm16.lit }
    val SYSCALL get() = sstms { IR.syscall.lit }
    val JUMP_ADDRESS get() = sstms { IR.jump_address.lit }

    val RS_IMM16 get() = sstms { RS + S_IMM16 }
    val real_jump_address get() = (ii.PC and (-268435456)) or IR.jump_address


    val S_IMM16_V: Int get() = sstms { IR.s_imm16 }
    val U_IMM16_V: Int get() = sstms { IR.u_imm16 }

    //fun set_pc(value: Int) {
    //    sstms.run {
    //        SET(p0[CpuState::_PC], value.lit)
    //        SET(p0[CpuState::_nPC], value.lit)
    //    }
    //}

    inline operator fun InstructionInfo.invoke(callback: D2Builder.(InstructionInfo) -> Unit): Unit {
        callback(sstms, this)
    }

    inline fun InstructionInfo.checkNan(callback: D2Builder.(InstructionInfo) -> Unit): Unit {
        callback(sstms, this)
        sstms.run {
            STM(::dyna_checkFNan.invoke(EXTERNAL, FD))
        }
    }

    inline fun InstructionInfo.preadvanceAndFlow(callback: D2Builder.(InstructionInfo) -> Unit): Unit {
        if (delayed == null) {
            incrementPC()
        }
        callback(sstms, this)
        reachedFlow = true
    }

    fun incrementPC() = sstms {
        //PC = (ii.PC + 4).lit
        //this.PC = sstms { (ii.PC + 4).lit }
        _PC = (ii.PC + 4).lit
        _nPC = (ii.PC + 8).lit
    }

    data class Delayed(val cond: D2ExprB?, val jumpTo: D2ExprI, val after: D2Stm?, val likely: Boolean)

    @PublishedApi
    internal var delayed: Delayed? = null

    inline fun InstructionInfo.branch(callback: D2Builder.(InstructionInfo) -> D2ExprI): Unit =
        _branch(likely = false, callback = callback)

    inline fun InstructionInfo.branchLikely(callback: D2Builder.(InstructionInfo) -> D2ExprI): Unit =
        _branch(likely = true, callback = callback)

    inline fun InstructionInfo._branch(
        likely: Boolean = false,
        callback: D2Builder.(InstructionInfo) -> D2ExprI
    ): Unit {
        sstms.run {
            delayed = Delayed(callback(this, this@_branch), (ii.PC + IR.s_imm16 * 4 + 4).lit, null, likely = likely)
        }
    }

    inline fun InstructionInfo.jump(
        noinline exec: (D2Builder.() -> Unit)? = null,
        callback: D2Builder.(InstructionInfo) -> D2ExprI
    ): Unit {
        sstms.run {
            val extraStm = if (exec != null) {
                D2Builder().useTemp {
                    exec()
                }.build()
            } else {
                null
            }
            delayed = Delayed(null, callback(this, this@jump), extraStm, likely = false)
        }
    }
}

fun D2Func.generateCpuStateFunction(ctx: D2Context, name: String? = null, debug: Boolean = false): (CpuState) -> Unit {
    return generate(ctx ,name, debug).generateCpuStateFunction()
}

fun D2Result.generateCpuStateFunction(): (CpuState) -> Unit {
    val rfunc = this.func
    return { cpu ->
        //println("Executing function... ${this.name}")
        //cpu.dump()
        val result = rfunc(cpu.registers.data.mem, cpu.mem, null, cpu)
        //println(" --> result=$result")
        if (result != 0) {
            throw CpuBreakExceptionCached(result)
        }
    }
}
