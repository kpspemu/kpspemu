package com.soywiz.kpspemu.cpu.interpreter

import com.soywiz.korio.lang.format
import com.soywiz.korio.util.*
import com.soywiz.kpspemu.cpu.*
import com.soywiz.kpspemu.cpu.dis.disasmMacro
import com.soywiz.kpspemu.util.BitUtils
import com.soywiz.kpspemu.util.imul32_64
import com.soywiz.kpspemu.util.umul32_64
import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.round

class CpuInterpreter(var cpu: CpuState, var trace: Boolean = false) {
	val dispatcher = InstructionDispatcher(InstructionInterpreter)

	fun step() {
		if (cpu._PC == 0) throw IllegalStateException("Trying to execute PC=0")
		if (trace) println("%08X: %s".format(cpu._PC, cpu.mem.disasmMacro(cpu._PC)))
		dispatcher.dispatch(cpu)
	}

	fun steps(count: Int) {
		for (n in 0 until count) step()
	}
}

// http://www.mrc.uidaho.edu/mrc/people/jff/digital/MIPSir.html
object InstructionInterpreter : InstructionEvaluator<CpuState>() {
	override fun unimplemented(s: CpuState, i: InstructionType): Unit = TODO("unimplemented: ${i.name} : " + i + " at ${"%08X".format(s._PC)}")

	val itemp = IntArray(2)

	// ALU
	override fun lui(s: CpuState) = s { RT = (U_IMM16 shl 16) }

	override fun movz(s: CpuState) = s { if (RT == 0) RD = RS }
	override fun movn(s: CpuState) = s { if (RT != 0) RD = RS }

	override fun ext(s: CpuState) = s { RT = RS.extract(POS, SIZE_E) }
	override fun ins(s: CpuState) = s { RT = RT.insert(RS, POS, SIZE_I) }

	override fun clz(s: CpuState) = s { RD = BitUtils.clz(RS) }
	override fun clo(s: CpuState) = s { RD = BitUtils.clo(RS) }
	override fun seb(s: CpuState) = s { RD = BitUtils.seb(RT) }
	override fun seh(s: CpuState) = s { RD = BitUtils.seh(RT) }

	override fun wsbh(s: CpuState) = s { RD = BitUtils.wsbh(RT) }
	override fun wsbw(s: CpuState) = s { RD = BitUtils.wsbw(RT) }

	override fun max(s: CpuState) = s { RD = kotlin.math.max(RS, RT) }
	override fun min(s: CpuState) = s { RD = kotlin.math.min(RS, RT) }

	override fun add(s: CpuState) = s { RD = RS + RT }
	override fun addu(s: CpuState) = s { RD = RS + RT }
	override fun sub(s: CpuState) = s { RD = RS - RT }
	override fun subu(s: CpuState) = s { RD = RS - RT }
	override fun addi(s: CpuState) = s { RT = RS + S_IMM16 }
	override fun addiu(s: CpuState) = s { RT = RS + S_IMM16 }

	override fun div(s: CpuState) = s { LO = RS / RT; HI = RS % RT }
	override fun divu(s: CpuState) = s { LO = RS udiv RT; HI = RS urem RT }

	override fun mult(s: CpuState) = s {
		imul32_64(RS, RT, itemp)
		this.LO = itemp[0]
		this.HI = itemp[1]
	}

	override fun multu(s: CpuState) = s {
		umul32_64(RS, RT, itemp)
		this.LO = itemp[0]
		this.HI = itemp[1]
	}

	override fun mflo(s: CpuState) = s { RD = LO }
	override fun mfhi(s: CpuState) = s { RD = HI }
	override fun mfic(s: CpuState) = s { RT = IC }

	override fun mtlo(s: CpuState) = s { LO = RS }
	override fun mthi(s: CpuState) = s { HI = RS }
	override fun mtic(s: CpuState) = s { IC = RT }

	// ALU: Bit
	override fun or(s: CpuState) = s { RD = RS or RT }

	override fun xor(s: CpuState) = s { RD = RS xor RT }
	override fun and(s: CpuState) = s { RD = RS and RT }
	override fun nor(s: CpuState) = s { RD = (RS or RT).inv() }

	override fun ori(s: CpuState) = s { RT = RS or U_IMM16 }
	override fun xori(s: CpuState) = s { RT = RS xor U_IMM16 }
	override fun andi(s: CpuState) = s { RT = RS and U_IMM16 }

	override fun sll(s: CpuState) = s { RD = RT shl POS }
	override fun sra(s: CpuState) = s { RD = RT shr POS }
	override fun srl(s: CpuState) = s { RD = RT ushr POS }

	override fun sllv(s: CpuState) = s { RD = RT shl (RS and 0b11111) }
	override fun srav(s: CpuState) = s { RD = RT shr (RS and 0b11111) }
	override fun srlv(s: CpuState) = s { RD = RT ushr (RS and 0b11111) }

	// Memory
	override fun lb(s: CpuState) = s { RT = mem.lb(RS_IMM16) }

	override fun lbu(s: CpuState) = s { RT = mem.lbu(RS_IMM16) }
	override fun lh(s: CpuState) = s { RT = mem.lh(RS_IMM16) }
	override fun lhu(s: CpuState) = s { RT = mem.lhu(RS_IMM16) }
	override fun lw(s: CpuState) = s { RT = mem.lw(RS_IMM16) }

	override fun sb(s: CpuState) = s { mem.sb(RS_IMM16, RT) }
	override fun sh(s: CpuState) = s { mem.sh(RS_IMM16, RT) }
	override fun sw(s: CpuState) = s { mem.sw(RS_IMM16, RT) }

	override fun lwc1(s: CpuState) = s { FT_I = mem.lw(RS_IMM16) }
	override fun swc1(s: CpuState) = s { mem.sw(RS_IMM16, FT_I) }

	// Special
	override fun syscall(s: CpuState) = s.preadvance { syscall(SYSCALL) }

	override fun _break(s: CpuState) = s.preadvance { throw CpuBreak(SYSCALL) }

	// Set less
	override fun slt(s: CpuState) = s { RD = if (IntEx.compare(RS, RT) < 0) 1 else 0 }

	override fun sltu(s: CpuState) = s { RD = if (IntEx.compareUnsigned(RS, RT) < 0) 1 else 0 }

	override fun slti(s: CpuState) = s { RT = if (IntEx.compare(RS, S_IMM16) < 0) 1 else 0 }
	override fun sltiu(s: CpuState) = s { RT = if (IntEx.compareUnsigned(RS, S_IMM16) < 0) 1 else 0 }


	// Branch
	override fun beq(s: CpuState) = s.branch { RS == RT }

	override fun bne(s: CpuState) = s.branch { RS != RT }
	override fun bltz(s: CpuState) = s.branch { RS < 0 }
	override fun blez(s: CpuState) = s.branch { RS <= 0 }
	override fun bgtz(s: CpuState) = s.branch { RS > 0 }
	override fun bgez(s: CpuState) = s.branch { RS >= 0 }

	override fun beql(s: CpuState) = s.branchLikely { RS == RT }
	override fun bnel(s: CpuState) = s.branchLikely { RS != RT }
	override fun bltzl(s: CpuState) = s.branchLikely { RS < 0 }
	override fun blezl(s: CpuState) = s.branchLikely { RS <= 0 }
	override fun bgtzl(s: CpuState) = s.branchLikely { RS > 0 }
	override fun bgezl(s: CpuState) = s.branchLikely { RS >= 0 }

	override fun j(s: CpuState) = s.none { _PC = _nPC; _nPC = (_PC and 0xf0000000.toInt()) or (JUMP_ADDRESS) }
	override fun jr(s: CpuState) = s.none { _PC = _nPC; _nPC = RS }

	// $31 = PC + 8 (or nPC + 4); PC = nPC; nPC = (PC & 0xf0000000) | (target << 2);
	override fun jal(s: CpuState) = s.none { RA = _nPC + 4; j(s) }

	override fun jalr(s: CpuState) = s.none { RA = _nPC + 4; jr(s) }

	// Float
	override fun mfc1(s: CpuState) = s { RT = FS_I }

	override fun mtc1(s: CpuState) = s { FS_I = RT }
	override fun cvt_s_w(s: CpuState) = s { FD = FS_I.toFloat() }
	override fun cvt_w_s(s: CpuState) = s {
		// @TODO: _cvt_w_s_impl, fcr31_rm: 0:rint, 1:cast, 2:ceil, 3:floor
		FD_I = FS.toInt()
	}

	override fun trunc_w_s(s: CpuState) = s { FD_I = FS.toInt() }
	override fun round_w_s(s: CpuState) = s { FD_I = round(FS).toInt() }
	override fun ceil_w_s(s: CpuState) = s { FD_I = ceil(FS).toInt() }
	override fun floor_w_s(s: CpuState) = s { FD_I = floor(FS).toInt() }

	override fun mov_s(s: CpuState) = s { FD = FS }
	override fun add_s(s: CpuState) = s { FD = FS + FT }
	override fun sub_s(s: CpuState) = s { FD = FS - FT }
	override fun mul_s(s: CpuState) = s { FD = FS * FT }
	override fun div_s(s: CpuState) = s { FD = FS / FT }
	override fun neg_s(s: CpuState) = s { FD = -FS }
	override fun abs_s(s: CpuState) = s { FD = kotlin.math.abs(FS) }
	override fun sqrt_s(s: CpuState) = s { FD = kotlin.math.sqrt(FS) }

	// Missing
	override fun bitrev(s: CpuState) = unimplemented(s, Instructions.bitrev)
	override fun rotr(s: CpuState) = unimplemented(s, Instructions.rotr)
	override fun rotrv(s: CpuState) = unimplemented(s, Instructions.rotrv)
	override fun madd(s: CpuState): Unit = unimplemented(s, Instructions.madd)
	override fun maddu(s: CpuState): Unit = unimplemented(s, Instructions.maddu)
	override fun msub(s: CpuState): Unit = unimplemented(s, Instructions.msub)
	override fun msubu(s: CpuState): Unit = unimplemented(s, Instructions.msubu)
	override fun bgezal(s: CpuState): Unit = unimplemented(s, Instructions.bgezal)
	override fun bgezall(s: CpuState): Unit = unimplemented(s, Instructions.bgezall)
	override fun bltzal(s: CpuState): Unit = unimplemented(s, Instructions.bltzal)
	override fun bltzall(s: CpuState): Unit = unimplemented(s, Instructions.bltzall)
	override fun bc1f(s: CpuState): Unit = unimplemented(s, Instructions.bc1f)
	override fun bc1t(s: CpuState): Unit = unimplemented(s, Instructions.bc1t)
	override fun bc1fl(s: CpuState): Unit = unimplemented(s, Instructions.bc1fl)
	override fun bc1tl(s: CpuState): Unit = unimplemented(s, Instructions.bc1tl)
	override fun lwl(s: CpuState): Unit = unimplemented(s, Instructions.lwl)
	override fun lwr(s: CpuState): Unit = unimplemented(s, Instructions.lwr)
	override fun swl(s: CpuState): Unit = unimplemented(s, Instructions.swl)
	override fun swr(s: CpuState): Unit = unimplemented(s, Instructions.swr)
	override fun ll(s: CpuState): Unit = unimplemented(s, Instructions.ll)
	override fun sc(s: CpuState): Unit = unimplemented(s, Instructions.sc)
	override fun cfc1(s: CpuState): Unit = unimplemented(s, Instructions.cfc1)
	override fun ctc1(s: CpuState): Unit = unimplemented(s, Instructions.ctc1)
	override fun c_f_s(s: CpuState): Unit = unimplemented(s, Instructions.c_f_s)
	override fun c_un_s(s: CpuState): Unit = unimplemented(s, Instructions.c_un_s)
	override fun c_eq_s(s: CpuState): Unit = unimplemented(s, Instructions.c_eq_s)
	override fun c_ueq_s(s: CpuState): Unit = unimplemented(s, Instructions.c_ueq_s)
	override fun c_olt_s(s: CpuState): Unit = unimplemented(s, Instructions.c_olt_s)
	override fun c_ult_s(s: CpuState): Unit = unimplemented(s, Instructions.c_ult_s)
	override fun c_ole_s(s: CpuState): Unit = unimplemented(s, Instructions.c_ole_s)
	override fun c_ule_s(s: CpuState): Unit = unimplemented(s, Instructions.c_ule_s)
	override fun c_sf_s(s: CpuState): Unit = unimplemented(s, Instructions.c_sf_s)
	override fun c_ngle_s(s: CpuState): Unit = unimplemented(s, Instructions.c_ngle_s)
	override fun c_seq_s(s: CpuState): Unit = unimplemented(s, Instructions.c_seq_s)
	override fun c_ngl_s(s: CpuState): Unit = unimplemented(s, Instructions.c_ngl_s)
	override fun c_lt_s(s: CpuState): Unit = unimplemented(s, Instructions.c_lt_s)
	override fun c_nge_s(s: CpuState): Unit = unimplemented(s, Instructions.c_nge_s)
	override fun c_le_s(s: CpuState): Unit = unimplemented(s, Instructions.c_le_s)
	override fun c_ngt_s(s: CpuState): Unit = unimplemented(s, Instructions.c_ngt_s)
	override fun cache(s: CpuState): Unit = unimplemented(s, Instructions.cache)
	override fun sync(s: CpuState): Unit = unimplemented(s, Instructions.sync)
	override fun dbreak(s: CpuState): Unit = unimplemented(s, Instructions.dbreak)
	override fun halt(s: CpuState): Unit = unimplemented(s, Instructions.halt)
	override fun dret(s: CpuState): Unit = unimplemented(s, Instructions.dret)
	override fun eret(s: CpuState): Unit = unimplemented(s, Instructions.eret)
	override fun mfdr(s: CpuState): Unit = unimplemented(s, Instructions.mfdr)
	override fun mtdr(s: CpuState): Unit = unimplemented(s, Instructions.mtdr)
	override fun cfc0(s: CpuState): Unit = unimplemented(s, Instructions.cfc0)
	override fun ctc0(s: CpuState): Unit = unimplemented(s, Instructions.ctc0)
	override fun mfc0(s: CpuState): Unit = unimplemented(s, Instructions.mfc0)
	override fun mtc0(s: CpuState): Unit = unimplemented(s, Instructions.mtc0)
	override fun mfv(s: CpuState): Unit = unimplemented(s, Instructions.mfv)
	override fun mfvc(s: CpuState): Unit = unimplemented(s, Instructions.mfvc)
	override fun mtv(s: CpuState): Unit = unimplemented(s, Instructions.mtv)
	override fun mtvc(s: CpuState): Unit = unimplemented(s, Instructions.mtvc)
	override fun lv_s(s: CpuState): Unit = unimplemented(s, Instructions.lv_s)
	override fun lv_q(s: CpuState): Unit = unimplemented(s, Instructions.lv_q)
	override fun lvl_q(s: CpuState): Unit = unimplemented(s, Instructions.lvl_q)
	override fun lvr_q(s: CpuState): Unit = unimplemented(s, Instructions.lvr_q)
	override fun sv_q(s: CpuState): Unit = unimplemented(s, Instructions.sv_q)
	override fun vdot(s: CpuState): Unit = unimplemented(s, Instructions.vdot)
	override fun vscl(s: CpuState): Unit = unimplemented(s, Instructions.vscl)
	override fun vsge(s: CpuState): Unit = unimplemented(s, Instructions.vsge)
	override fun vslt(s: CpuState): Unit = unimplemented(s, Instructions.vslt)
	override fun vrot(s: CpuState): Unit = unimplemented(s, Instructions.vrot)
	override fun vzero(s: CpuState): Unit = unimplemented(s, Instructions.vzero)
	override fun vone(s: CpuState): Unit = unimplemented(s, Instructions.vone)
	override fun vmov(s: CpuState): Unit = unimplemented(s, Instructions.vmov)
	override fun vabs(s: CpuState): Unit = unimplemented(s, Instructions.vabs)
	override fun vneg(s: CpuState): Unit = unimplemented(s, Instructions.vneg)
	override fun vocp(s: CpuState): Unit = unimplemented(s, Instructions.vocp)
	override fun vsgn(s: CpuState): Unit = unimplemented(s, Instructions.vsgn)
	override fun vrcp(s: CpuState): Unit = unimplemented(s, Instructions.vrcp)
	override fun vrsq(s: CpuState): Unit = unimplemented(s, Instructions.vrsq)
	override fun vsin(s: CpuState): Unit = unimplemented(s, Instructions.vsin)
	override fun vcos(s: CpuState): Unit = unimplemented(s, Instructions.vcos)
	override fun vexp2(s: CpuState): Unit = unimplemented(s, Instructions.vexp2)
	override fun vlog2(s: CpuState): Unit = unimplemented(s, Instructions.vlog2)
	override fun vsqrt(s: CpuState): Unit = unimplemented(s, Instructions.vsqrt)
	override fun vasin(s: CpuState): Unit = unimplemented(s, Instructions.vasin)
	override fun vnrcp(s: CpuState): Unit = unimplemented(s, Instructions.vnrcp)
	override fun vnsin(s: CpuState): Unit = unimplemented(s, Instructions.vnsin)
	override fun vrexp2(s: CpuState): Unit = unimplemented(s, Instructions.vrexp2)
	override fun vsat0(s: CpuState): Unit = unimplemented(s, Instructions.vsat0)
	override fun vsat1(s: CpuState): Unit = unimplemented(s, Instructions.vsat1)
	override fun vcst(s: CpuState): Unit = unimplemented(s, Instructions.vcst)
	override fun vmmul(s: CpuState): Unit = unimplemented(s, Instructions.vmmul)
	override fun vhdp(s: CpuState): Unit = unimplemented(s, Instructions.vhdp)
	override fun vcrs_t(s: CpuState): Unit = unimplemented(s, Instructions.vcrs_t)
	override fun vcrsp_t(s: CpuState): Unit = unimplemented(s, Instructions.vcrsp_t)
	override fun vi2c(s: CpuState): Unit = unimplemented(s, Instructions.vi2c)
	override fun vi2uc(s: CpuState): Unit = unimplemented(s, Instructions.vi2uc)
	override fun vtfm2(s: CpuState): Unit = unimplemented(s, Instructions.vtfm2)
	override fun vtfm3(s: CpuState): Unit = unimplemented(s, Instructions.vtfm3)
	override fun vtfm4(s: CpuState): Unit = unimplemented(s, Instructions.vtfm4)
	override fun vhtfm2(s: CpuState): Unit = unimplemented(s, Instructions.vhtfm2)
	override fun vhtfm3(s: CpuState): Unit = unimplemented(s, Instructions.vhtfm3)
	override fun vhtfm4(s: CpuState): Unit = unimplemented(s, Instructions.vhtfm4)
	override fun vsrt3(s: CpuState): Unit = unimplemented(s, Instructions.vsrt3)
	override fun vfad(s: CpuState): Unit = unimplemented(s, Instructions.vfad)
	override fun vmin(s: CpuState): Unit = unimplemented(s, Instructions.vmin)
	override fun vmax(s: CpuState): Unit = unimplemented(s, Instructions.vmax)
	override fun vadd(s: CpuState): Unit = unimplemented(s, Instructions.vadd)
	override fun vsub(s: CpuState): Unit = unimplemented(s, Instructions.vsub)
	override fun vdiv(s: CpuState): Unit = unimplemented(s, Instructions.vdiv)
	override fun vmul(s: CpuState): Unit = unimplemented(s, Instructions.vmul)
	override fun vidt(s: CpuState): Unit = unimplemented(s, Instructions.vidt)
	override fun vmidt(s: CpuState): Unit = unimplemented(s, Instructions.vmidt)
	override fun viim(s: CpuState): Unit = unimplemented(s, Instructions.viim)
	override fun vmmov(s: CpuState): Unit = unimplemented(s, Instructions.vmmov)
	override fun vmzero(s: CpuState): Unit = unimplemented(s, Instructions.vmzero)
	override fun vmone(s: CpuState): Unit = unimplemented(s, Instructions.vmone)
	override fun vnop(s: CpuState): Unit = unimplemented(s, Instructions.vnop)
	override fun vsync(s: CpuState): Unit = unimplemented(s, Instructions.vsync)
	override fun vflush(s: CpuState): Unit = unimplemented(s, Instructions.vflush)
	override fun vpfxd(s: CpuState): Unit = unimplemented(s, Instructions.vpfxd)
	override fun vpfxs(s: CpuState): Unit = unimplemented(s, Instructions.vpfxs)
	override fun vpfxt(s: CpuState): Unit = unimplemented(s, Instructions.vpfxt)
	override fun vdet(s: CpuState): Unit = unimplemented(s, Instructions.vdet)
	override fun vrnds(s: CpuState): Unit = unimplemented(s, Instructions.vrnds)
	override fun vrndi(s: CpuState): Unit = unimplemented(s, Instructions.vrndi)
	override fun vrndf1(s: CpuState): Unit = unimplemented(s, Instructions.vrndf1)
	override fun vrndf2(s: CpuState): Unit = unimplemented(s, Instructions.vrndf2)
	override fun vcmp(s: CpuState): Unit = unimplemented(s, Instructions.vcmp)
	override fun vcmovf(s: CpuState): Unit = unimplemented(s, Instructions.vcmovf)
	override fun vcmovt(s: CpuState): Unit = unimplemented(s, Instructions.vcmovt)
	override fun vavg(s: CpuState): Unit = unimplemented(s, Instructions.vavg)
	override fun vf2id(s: CpuState): Unit = unimplemented(s, Instructions.vf2id)
	override fun vf2in(s: CpuState): Unit = unimplemented(s, Instructions.vf2in)
	override fun vf2iu(s: CpuState): Unit = unimplemented(s, Instructions.vf2iu)
	override fun vf2iz(s: CpuState): Unit = unimplemented(s, Instructions.vf2iz)
	override fun vi2f(s: CpuState): Unit = unimplemented(s, Instructions.vi2f)
	override fun vscmp(s: CpuState): Unit = unimplemented(s, Instructions.vscmp)
	override fun vmscl(s: CpuState): Unit = unimplemented(s, Instructions.vmscl)
	override fun vt4444_q(s: CpuState): Unit = unimplemented(s, Instructions.vt4444_q)
	override fun vt5551_q(s: CpuState): Unit = unimplemented(s, Instructions.vt5551_q)
	override fun vt5650_q(s: CpuState): Unit = unimplemented(s, Instructions.vt5650_q)
	override fun vmfvc(s: CpuState): Unit = unimplemented(s, Instructions.vmfvc)
	override fun vmtvc(s: CpuState): Unit = unimplemented(s, Instructions.vmtvc)
	override fun mfvme(s: CpuState): Unit = unimplemented(s, Instructions.mfvme)
	override fun mtvme(s: CpuState): Unit = unimplemented(s, Instructions.mtvme)
	override fun sv_s(s: CpuState): Unit = unimplemented(s, Instructions.sv_s)
	override fun vfim(s: CpuState): Unit = unimplemented(s, Instructions.vfim)
	override fun svl_q(s: CpuState): Unit = unimplemented(s, Instructions.svl_q)
	override fun svr_q(s: CpuState): Unit = unimplemented(s, Instructions.svr_q)
	override fun vbfy1(s: CpuState): Unit = unimplemented(s, Instructions.vbfy1)
	override fun vbfy2(s: CpuState): Unit = unimplemented(s, Instructions.vbfy2)
	override fun vf2h(s: CpuState): Unit = unimplemented(s, Instructions.vf2h)
	override fun vh2f(s: CpuState): Unit = unimplemented(s, Instructions.vh2f)
	override fun vi2s(s: CpuState): Unit = unimplemented(s, Instructions.vi2s)
	override fun vi2us(s: CpuState): Unit = unimplemented(s, Instructions.vi2us)
	override fun vlgb(s: CpuState): Unit = unimplemented(s, Instructions.vlgb)
	override fun vqmul(s: CpuState): Unit = unimplemented(s, Instructions.vqmul)
	override fun vs2i(s: CpuState): Unit = unimplemented(s, Instructions.vs2i)
	override fun vc2i(s: CpuState): Unit = unimplemented(s, Instructions.vc2i)
	override fun vuc2i(s: CpuState): Unit = unimplemented(s, Instructions.vuc2i)
	override fun vsbn(s: CpuState): Unit = unimplemented(s, Instructions.vsbn)
	override fun vsbz(s: CpuState): Unit = unimplemented(s, Instructions.vsbz)
	override fun vsocp(s: CpuState): Unit = unimplemented(s, Instructions.vsocp)
	override fun vsrt1(s: CpuState): Unit = unimplemented(s, Instructions.vsrt1)
	override fun vsrt2(s: CpuState): Unit = unimplemented(s, Instructions.vsrt2)
	override fun vsrt4(s: CpuState): Unit = unimplemented(s, Instructions.vsrt4)
	override fun vus2i(s: CpuState): Unit = unimplemented(s, Instructions.vus2i)
	override fun vwbn(s: CpuState): Unit = unimplemented(s, Instructions.vwbn)
	override fun bvf(s: CpuState): Unit = unimplemented(s, Instructions.bvf)
	override fun bvt(s: CpuState): Unit = unimplemented(s, Instructions.bvt)
	override fun bvfl(s: CpuState): Unit = unimplemented(s, Instructions.bvfl)
	override fun bvtl(s: CpuState): Unit = unimplemented(s, Instructions.bvtl)
}

