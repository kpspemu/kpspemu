package com.soywiz.kpspemu.cpu.interpreter

import com.soywiz.kmem.FastMemory
import com.soywiz.kmem.get
import com.soywiz.korio.error.invalidOp
import com.soywiz.korio.lang.Console
import com.soywiz.korio.lang.format
import com.soywiz.korio.util.*
import com.soywiz.korma.math.Math
import com.soywiz.korma.math.isAlmostZero
import com.soywiz.kpspemu.cpu.*
import com.soywiz.kpspemu.cpu.dis.disasmMacro
import com.soywiz.kpspemu.mem.Memory
import com.soywiz.kpspemu.util.FloatArray2
import com.soywiz.kpspemu.util.cosv1
import com.soywiz.kpspemu.util.sinv1
import kotlin.math.*

class CpuInterpreter(var cpu: CpuState, val breakpoints: Breakpoints, var trace: Boolean = false) {
	val dispatcher = InstructionDispatcher(InstructionInterpreter)

	fun steps(count: Int): Int {
		val mem = cpu.mem.getFastMem()
		//val mem = null
		return if (mem != null) {
			stepsFastMem(mem, cpu.mem.getFastMemOffset(Memory.MAIN_OFFSET) - Memory.MAIN_OFFSET, count)
		} else {
			stepsNormal(count)
		}
	}

	fun stepsNormal(count: Int): Int {
		val dispatcher = this.dispatcher
		val cpu = this.cpu
		val mem = cpu.mem
		val trace = this.trace
		var sPC = 0
		var n = 0
		//val fast = (mem as FastMemory).buffer
		val breakpointsEnabled = breakpoints.enabled
		try {
			while (n < count) {
				sPC = cpu._PC
				if (breakpointsEnabled && breakpoints[sPC]) throw BreakpointException(cpu, sPC)
				n++
				//if (PC == 0) throw IllegalStateException("Trying to execute PC=0")
				if (trace) tracePC()
				val IR = mem.lw(sPC)
				//val IR = fast.getAlignedInt32((PC ushr 2) and Memory.MASK)
				cpu.IR = IR
				dispatcher.dispatch(cpu, sPC, IR)
			}
		} catch (e: Throwable) {
			if (e !is CpuBreakException) {
				Console.error("There was an error at %08X: %s".format(sPC, cpu.mem.disasmMacro(sPC)))
			}
			throw e
		} finally {
			cpu.totalExecuted += n
		}
		return n
	}

	fun stepsFastMem(mem: FastMemory, memOffset: Int, count: Int): Int {
		val i32 = mem.i32
		val cpu = this.cpu
		var n = 0
		var sPC = 0
		val breakpointsEnabled = breakpoints.enabled
		try {
			while (n < count) {
				sPC = cpu._PC and 0x0FFFFFFF
				if (breakpointsEnabled && breakpoints[sPC]) throw BreakpointException(cpu, sPC)
				n++
				val IR = i32[(sPC + memOffset) ushr 2]
				cpu.IR = IR
				dispatcher.dispatch(cpu, sPC, IR)
			}
		} catch (e: Throwable) {
			if (e !is CpuBreakException) {
				Console.error("There was an error at %08X: %s".format(sPC, cpu.mem.disasmMacro(sPC)))
			}
			throw e
		} finally {
			cpu.totalExecuted += n
		}
		return n
	}

	private fun tracePC() {
		println("%08X: %s".format(cpu._PC, cpu.mem.disasmMacro(cpu._PC)))
	}
}

@Suppress("FunctionName")
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
	override fun divu(s: CpuState) = s {
		val d = RT
		if (d != 0) {
			LO = RS udiv d
			HI = RS urem d
		} else {
			LO = 0
			HI = 0
		}
	}

	override fun mult(s: CpuState) = s { imul32_64(RS, RT, itemp); this.LO = itemp[0]; this.HI = itemp[1] }
	override fun multu(s: CpuState) = s { umul32_64(RS, RT, itemp); this.LO = itemp[0]; this.HI = itemp[1] }

	override fun madd(s: CpuState) = s { HI_LO += RS.toLong() * RT.toLong() }
	override fun maddu(s: CpuState) = s { HI_LO += RS.unsigned * RT.unsigned }

	override fun msub(s: CpuState) = s { HI_LO -= RS.toLong() * RT.toLong() }
	override fun msubu(s: CpuState) = s { HI_LO -= RS.unsigned * RT.unsigned }


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

	override fun bitrev(s: CpuState) = s { RD = BitUtils.bitrev32(RT) }

	override fun rotr(s: CpuState) = s { RD = BitUtils.rotr(RT, POS) }
	override fun rotrv(s: CpuState) = s { RD = BitUtils.rotr(RT, RS) }

	// Memory
	override fun lb(s: CpuState) = s { RT = mem.lb(RS_IMM16) }

	override fun lbu(s: CpuState) = s { RT = mem.lbu(RS_IMM16) }
	override fun lh(s: CpuState) = s { RT = mem.lh(RS_IMM16) }
	override fun lhu(s: CpuState) = s { RT = mem.lhu(RS_IMM16) }
	override fun lw(s: CpuState) = s { RT = mem.lw(RS_IMM16) }

	override fun lwl(s: CpuState) = s { RT = mem.lwl(RS_IMM16, RT) }
	override fun lwr(s: CpuState) = s { RT = mem.lwr(RS_IMM16, RT) }

	override fun swl(s: CpuState) = s { mem.swl(RS_IMM16, RT) }
	override fun swr(s: CpuState) = s { mem.swr(RS_IMM16, RT) }

	override fun sb(s: CpuState) = s { mem.sb(RS_IMM16, RT) }
	override fun sh(s: CpuState) = s { mem.sh(RS_IMM16, RT) }
	override fun sw(s: CpuState) = s { mem.sw(RS_IMM16, RT) }

	override fun lwc1(s: CpuState) = s { FT_I = mem.lw(RS_IMM16) }
	override fun swc1(s: CpuState) = s { mem.sw(RS_IMM16, FT_I) }

	// Special
	override fun syscall(s: CpuState) = s.preadvance { syscall(SYSCALL) }

	override fun _break(s: CpuState) = s.preadvance { throw CpuBreakException(SYSCALL) }

	// Set less
	override fun slt(s: CpuState) = s { RD = (RS < RT).toInt() }

	override fun sltu(s: CpuState) = s { RD = (RS ult RT).toInt() }

	override fun slti(s: CpuState) = s { RT = (RS < S_IMM16).toInt() }
	override fun sltiu(s: CpuState) = s { RT = (RS ult S_IMM16).toInt() }


	// Branch
	override fun beq(s: CpuState) = s.branch { RS == RT }

	override fun bne(s: CpuState) = s.branch { RS != RT }
	override fun bltz(s: CpuState) = s.branch { RS < 0 }
	override fun blez(s: CpuState) = s.branch { RS <= 0 }
	override fun bgtz(s: CpuState) = s.branch { RS > 0 }
	override fun bgez(s: CpuState) = s.branch { RS >= 0 }
	override fun bgezal(s: CpuState) = s.branch { RA = _nPC + 4; RS >= 0 }
	override fun bltzal(s: CpuState) = s.branch { RA = _nPC + 4; RS < 0 }

	override fun beql(s: CpuState) = s.branchLikely { RS == RT }
	override fun bnel(s: CpuState) = s.branchLikely { RS != RT }
	override fun bltzl(s: CpuState) = s.branchLikely { RS < 0 }
	override fun blezl(s: CpuState) = s.branchLikely { RS <= 0 }
	override fun bgtzl(s: CpuState) = s.branchLikely { RS > 0 }
	override fun bgezl(s: CpuState) = s.branchLikely { RS >= 0 }
	override fun bgezall(s: CpuState) = s.branchLikely { RA = _nPC + 4; RS >= 0 }
	override fun bltzall(s: CpuState) = s.branchLikely { RA = _nPC + 4; RS < 0 }


	override fun bc1f(s: CpuState) = s.branch { !fcr31_cc }
	override fun bc1t(s: CpuState) = s.branch { fcr31_cc }
	override fun bc1fl(s: CpuState) = s.branchLikely { !fcr31_cc }
	override fun bc1tl(s: CpuState) = s.branchLikely { fcr31_cc }

	//override fun j(s: CpuState) = s.none { _PC = _nPC; _nPC = (_PC and 0xf0000000.toInt()) or (JUMP_ADDRESS) } // @TODO: Kotlin.JS doesn't optimize 0xf0000000.toInt() and generates a long
	override fun j(s: CpuState) = s.none { _PC = _nPC; _nPC = (_PC and (-268435456)) or (JUMP_ADDRESS) }

	override fun jr(s: CpuState) = s.none { _PC = _nPC; _nPC = RS }

	override fun jal(s: CpuState) = s.none { j(s); RA = _PC + 4; } // $31 = PC + 8 (or nPC + 4); PC = nPC; nPC = (PC & 0xf0000000) | (target << 2);
	override fun jalr(s: CpuState) = s.none { jr(s); RD = _PC + 4; }

	// Float
	override fun mfc1(s: CpuState) = s { RT = FS_I }

	override fun mtc1(s: CpuState) = s { FS_I = RT }
	override fun cvt_s_w(s: CpuState) = s { FD = FS_I.toFloat() }
	override fun cvt_w_s(s: CpuState) = s {

		FD_I = when (this.fcr31_rm) {
			0 -> Math.rint(FS) // rint: round nearest
			1 -> Math.cast(FS) // round to zero
			2 -> Math.ceil(FS) // round up (ceil)
			3 -> Math.floor(FS) // round down (floor)
			else -> FS.toInt()
		}
	}

	override fun trunc_w_s(s: CpuState) = s { FD_I = Math.trunc(FS) }
	override fun round_w_s(s: CpuState) = s { FD_I = Math.round(FS) }
	override fun ceil_w_s(s: CpuState) = s { FD_I = Math.ceil(FS) }
	override fun floor_w_s(s: CpuState) = s { FD_I = Math.floor(FS) }

	inline fun CpuState.checkNan(callback: CpuState.() -> Unit) = this.normal {
		callback()
		if (FD.isNaN()) fcr31 = fcr31 or 0x00010040
		if (FD.isInfinite()) fcr31 = fcr31 or 0x00005014
	}

	override fun mov_s(s: CpuState) = s.checkNan { FD = FS }
	override fun add_s(s: CpuState) = s.checkNan { FD = FS + FT }
	override fun sub_s(s: CpuState) = s.checkNan { FD = FS - FT }
	override fun mul_s(s: CpuState) = s.checkNan { FD = FS * FT; if (fcr31_fs && FD.isAlmostZero()) FD = 0f }
	override fun div_s(s: CpuState) = s.checkNan { FD = FS / FT }
	override fun neg_s(s: CpuState) = s.checkNan { FD = -FS }
	override fun abs_s(s: CpuState) = s.checkNan { FD = kotlin.math.abs(FS) }
	override fun sqrt_s(s: CpuState) = s.checkNan { FD = kotlin.math.sqrt(FS) }

	private inline fun CpuState._cu(callback: CpuState.() -> Boolean) = this { fcr31_cc = if (FS.isNaN() || FT.isNaN()) true else callback() }
	private inline fun CpuState._co(callback: CpuState.() -> Boolean) = this { fcr31_cc = if (FS.isNaN() || FT.isNaN()) false else callback() }

	override fun c_f_s(s: CpuState) = s._co { false }
	override fun c_un_s(s: CpuState) = s._cu { false }
	override fun c_eq_s(s: CpuState) = s._co { FS == FT }
	override fun c_ueq_s(s: CpuState) = s._cu { FS == FT }
	override fun c_olt_s(s: CpuState) = s._co { FS < FT }
	override fun c_ult_s(s: CpuState) = s._cu { FS < FT }
	override fun c_ole_s(s: CpuState) = s._co { FS <= FT }
	override fun c_ule_s(s: CpuState) = s._cu { FS <= FT }

	override fun c_sf_s(s: CpuState) = s._co { false }
	override fun c_ngle_s(s: CpuState) = s._cu { false }
	override fun c_seq_s(s: CpuState) = s._co { FS == FT }
	override fun c_ngl_s(s: CpuState) = s._cu { FS == FT }
	override fun c_lt_s(s: CpuState) = s._co { FS < FT }
	override fun c_nge_s(s: CpuState) = s._cu { FS < FT }
	override fun c_le_s(s: CpuState) = s._co { FS <= FT }
	override fun c_ngt_s(s: CpuState) = s._cu { FS <= FT }

	override fun cfc1(s: CpuState) = s {
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

	override fun ctc1(s: CpuState) = s {
		when (IR.rd) {
			31 -> updateFCR31(RT)
		}
	}

	val tempInts = IntArray(16)

	override fun lvl_q(s: CpuState) = s {
		vectorRegisters(IR.vt5_1, VectorSize.Quad) { i, r -> tempInts[i] = getVfprI(r) }
		mem.lvl_q(RS_IMM14, tempInts)
	}
	override fun lvr_q(s: CpuState) = s {
		vectorRegisters(IR.vt5_1, VectorSize.Quad) { i, r -> tempInts[i] = getVfprI(r) }
		mem.lvr_q(RS_IMM14, tempInts)
	}
	override fun sv_q(s: CpuState) = s { val start = IR.s_imm14; vectorRegisters(IR.vt5_1, VectorSize.Quad) { i, r -> mem.sw(start + i * 4, s.getVfprI(r)) } }
	override fun lv_q(s: CpuState) = s { val start = IR.s_imm14; vectorRegisters(IR.vt5_1, VectorSize.Quad) { i, r -> s.setVfprI(r, mem.lw(start + i * 4 + 4)) } }
	override fun viim(s: CpuState) = s { VT = S_IMM16.toFloat() }
	override fun vcst(s: CpuState) = s { VD = VfpuConstants[IR.imm5].value }
	override fun mtv(s: CpuState) = s { VD_I = RT }
	override fun vpfxt(s: CpuState) = s { vpfxt = IR; vpfxtEnabled = true }
	override fun vpfxd(s: CpuState) = s { vpfxd = IR; vpfxdEnabled = true }
	override fun vpfxs(s: CpuState) = s { vpfxs = IR; vpfxsEnabled = true }
	override fun vmov(s: CpuState) = s { setVD_VS { vs[it] } }
	override fun vrcp(s: CpuState) = s { setVD_VS { 1f / vs[it] } }
	override fun vmul(s: CpuState) = s { setVD_VSVT { vs[it] * vt[it] } }
	override fun vdiv(s: CpuState) = s { setVD_VSVT { vs[it] / vt[it] } }
	override fun vadd(s: CpuState) = s { setVD_VSVT { vs[it] + vt[it] } }
	override fun vsub(s: CpuState) = s { setVD_VSVT { vs[it] - vt[it] } }
	override fun vrot(s: CpuState) = s {
		val vectorSize = IR.one_two
		val imm5 = IR.imm5
		val cosIndex = imm5.extract(0, 2)
		val sinIndex = imm5.extract(2, 2)
		val negateSin = imm5.extractBool(4)

		setVD_VS(vectorSize, 1) {
			var sine = sinv1(vs[0])
			val cosine = cosv1(vs[0])
			if (negateSin) sine = -sine

			when (it) {
				cosIndex -> cosine
				sinIndex -> sine
				else -> if (sinIndex == cosIndex) sine else 0f
			}
		}
	}

	override fun vmzero(s: CpuState) = s { setMatrixVD { 0f } }
	override fun vmone(s: CpuState) = s { setMatrixVD { 1f } }
	override fun vmidt(s: CpuState) = s { setMatrixVD { if (row == col) 1f else 0f } }
	override fun vmmov(s: CpuState) = s { setMatrixVD_VS { ms[col, row] } }
	override fun vmmul(s: CpuState) = s { setMatrixVD_VSVT { (0 until side).map { ms[col, row] * mt[row, col] }.sum() } }

	// Missing

	override fun ll(s: CpuState) = unimplemented(s, Instructions.ll)
	override fun sc(s: CpuState) = unimplemented(s, Instructions.sc)

	override fun cache(s: CpuState) = unimplemented(s, Instructions.cache)
	override fun sync(s: CpuState) = unimplemented(s, Instructions.sync)
	override fun dbreak(s: CpuState) = unimplemented(s, Instructions.dbreak)
	override fun halt(s: CpuState) = unimplemented(s, Instructions.halt)
	override fun dret(s: CpuState) = unimplemented(s, Instructions.dret)
	override fun eret(s: CpuState) = unimplemented(s, Instructions.eret)
	override fun mfdr(s: CpuState) = unimplemented(s, Instructions.mfdr)
	override fun mtdr(s: CpuState) = unimplemented(s, Instructions.mtdr)
	override fun cfc0(s: CpuState) = unimplemented(s, Instructions.cfc0)
	override fun ctc0(s: CpuState) = unimplemented(s, Instructions.ctc0)
	override fun mfc0(s: CpuState) = unimplemented(s, Instructions.mfc0)
	override fun mtc0(s: CpuState) = unimplemented(s, Instructions.mtc0)
	override fun mfv(s: CpuState) = unimplemented(s, Instructions.mfv)
	override fun mfvc(s: CpuState) = unimplemented(s, Instructions.mfvc)
	override fun mtvc(s: CpuState) = unimplemented(s, Instructions.mtvc)
	override fun lv_s(s: CpuState) = unimplemented(s, Instructions.lv_s)
	override fun vdot(s: CpuState) = unimplemented(s, Instructions.vdot)
	override fun vscl(s: CpuState) = unimplemented(s, Instructions.vscl)
	override fun vsge(s: CpuState) = unimplemented(s, Instructions.vsge)
	override fun vslt(s: CpuState) = unimplemented(s, Instructions.vslt)
	override fun vzero(s: CpuState) = unimplemented(s, Instructions.vzero)
	override fun vone(s: CpuState) = unimplemented(s, Instructions.vone)
	override fun vabs(s: CpuState) = unimplemented(s, Instructions.vabs)
	override fun vneg(s: CpuState) = unimplemented(s, Instructions.vneg)
	override fun vocp(s: CpuState) = unimplemented(s, Instructions.vocp)
	override fun vsgn(s: CpuState) = unimplemented(s, Instructions.vsgn)
	override fun vrsq(s: CpuState) = unimplemented(s, Instructions.vrsq)
	override fun vsin(s: CpuState) = unimplemented(s, Instructions.vsin)
	override fun vcos(s: CpuState) = unimplemented(s, Instructions.vcos)
	override fun vexp2(s: CpuState) = unimplemented(s, Instructions.vexp2)
	override fun vlog2(s: CpuState) = unimplemented(s, Instructions.vlog2)
	override fun vsqrt(s: CpuState) = unimplemented(s, Instructions.vsqrt)
	override fun vasin(s: CpuState) = unimplemented(s, Instructions.vasin)
	override fun vnrcp(s: CpuState) = unimplemented(s, Instructions.vnrcp)
	override fun vnsin(s: CpuState) = unimplemented(s, Instructions.vnsin)
	override fun vrexp2(s: CpuState) = unimplemented(s, Instructions.vrexp2)
	override fun vsat0(s: CpuState) = unimplemented(s, Instructions.vsat0)
	override fun vsat1(s: CpuState) = unimplemented(s, Instructions.vsat1)
	override fun vhdp(s: CpuState) = unimplemented(s, Instructions.vhdp)
	override fun vcrs_t(s: CpuState) = unimplemented(s, Instructions.vcrs_t)
	override fun vcrsp_t(s: CpuState) = unimplemented(s, Instructions.vcrsp_t)
	override fun vi2c(s: CpuState) = unimplemented(s, Instructions.vi2c)
	override fun vi2uc(s: CpuState) = unimplemented(s, Instructions.vi2uc)
	override fun vtfm2(s: CpuState) = unimplemented(s, Instructions.vtfm2)
	override fun vtfm3(s: CpuState) = unimplemented(s, Instructions.vtfm3)
	override fun vtfm4(s: CpuState) = unimplemented(s, Instructions.vtfm4)
	override fun vhtfm2(s: CpuState) = unimplemented(s, Instructions.vhtfm2)
	override fun vhtfm3(s: CpuState) = unimplemented(s, Instructions.vhtfm3)
	override fun vhtfm4(s: CpuState) = unimplemented(s, Instructions.vhtfm4)
	override fun vsrt3(s: CpuState) = unimplemented(s, Instructions.vsrt3)
	override fun vfad(s: CpuState) = unimplemented(s, Instructions.vfad)
	override fun vmin(s: CpuState) = unimplemented(s, Instructions.vmin)
	override fun vmax(s: CpuState) = unimplemented(s, Instructions.vmax)
	override fun vidt(s: CpuState) = unimplemented(s, Instructions.vidt)
	override fun vnop(s: CpuState) = unimplemented(s, Instructions.vnop)
	override fun vsync(s: CpuState) = unimplemented(s, Instructions.vsync)
	override fun vflush(s: CpuState) = unimplemented(s, Instructions.vflush)
	override fun vdet(s: CpuState) = unimplemented(s, Instructions.vdet)
	override fun vrnds(s: CpuState) = unimplemented(s, Instructions.vrnds)
	override fun vrndi(s: CpuState) = unimplemented(s, Instructions.vrndi)
	override fun vrndf1(s: CpuState) = unimplemented(s, Instructions.vrndf1)
	override fun vrndf2(s: CpuState) = unimplemented(s, Instructions.vrndf2)
	override fun vcmp(s: CpuState) = unimplemented(s, Instructions.vcmp)
	override fun vcmovf(s: CpuState) = unimplemented(s, Instructions.vcmovf)
	override fun vcmovt(s: CpuState) = unimplemented(s, Instructions.vcmovt)
	override fun vavg(s: CpuState) = unimplemented(s, Instructions.vavg)
	override fun vf2id(s: CpuState) = unimplemented(s, Instructions.vf2id)
	override fun vf2in(s: CpuState) = unimplemented(s, Instructions.vf2in)
	override fun vf2iu(s: CpuState) = unimplemented(s, Instructions.vf2iu)
	override fun vf2iz(s: CpuState) = unimplemented(s, Instructions.vf2iz)
	override fun vi2f(s: CpuState) = unimplemented(s, Instructions.vi2f)
	override fun vscmp(s: CpuState) = unimplemented(s, Instructions.vscmp)
	override fun vmscl(s: CpuState) = unimplemented(s, Instructions.vmscl)
	override fun vt4444_q(s: CpuState) = unimplemented(s, Instructions.vt4444_q)
	override fun vt5551_q(s: CpuState) = unimplemented(s, Instructions.vt5551_q)
	override fun vt5650_q(s: CpuState) = unimplemented(s, Instructions.vt5650_q)
	override fun vmfvc(s: CpuState) = unimplemented(s, Instructions.vmfvc)
	override fun vmtvc(s: CpuState) = unimplemented(s, Instructions.vmtvc)
	override fun mfvme(s: CpuState) = unimplemented(s, Instructions.mfvme)
	override fun mtvme(s: CpuState) = unimplemented(s, Instructions.mtvme)
	override fun sv_s(s: CpuState) = unimplemented(s, Instructions.sv_s)
	override fun vfim(s: CpuState) = unimplemented(s, Instructions.vfim)
	override fun svl_q(s: CpuState) = unimplemented(s, Instructions.svl_q)
	override fun svr_q(s: CpuState) = unimplemented(s, Instructions.svr_q)
	override fun vbfy1(s: CpuState) = unimplemented(s, Instructions.vbfy1)
	override fun vbfy2(s: CpuState) = unimplemented(s, Instructions.vbfy2)
	override fun vf2h(s: CpuState) = unimplemented(s, Instructions.vf2h)
	override fun vh2f(s: CpuState) = unimplemented(s, Instructions.vh2f)
	override fun vi2s(s: CpuState) = unimplemented(s, Instructions.vi2s)
	override fun vi2us(s: CpuState) = unimplemented(s, Instructions.vi2us)
	override fun vlgb(s: CpuState) = unimplemented(s, Instructions.vlgb)
	override fun vqmul(s: CpuState) = unimplemented(s, Instructions.vqmul)
	override fun vs2i(s: CpuState) = unimplemented(s, Instructions.vs2i)
	override fun vc2i(s: CpuState) = unimplemented(s, Instructions.vc2i)
	override fun vuc2i(s: CpuState) = unimplemented(s, Instructions.vuc2i)
	override fun vsbn(s: CpuState) = unimplemented(s, Instructions.vsbn)
	override fun vsbz(s: CpuState) = unimplemented(s, Instructions.vsbz)
	override fun vsocp(s: CpuState) = unimplemented(s, Instructions.vsocp)
	override fun vsrt1(s: CpuState) = unimplemented(s, Instructions.vsrt1)
	override fun vsrt2(s: CpuState) = unimplemented(s, Instructions.vsrt2)
	override fun vsrt4(s: CpuState) = unimplemented(s, Instructions.vsrt4)
	override fun vus2i(s: CpuState) = unimplemented(s, Instructions.vus2i)
	override fun vwbn(s: CpuState) = unimplemented(s, Instructions.vwbn)
	override fun bvf(s: CpuState) = unimplemented(s, Instructions.bvf)
	override fun bvt(s: CpuState) = unimplemented(s, Instructions.bvt)
	override fun bvfl(s: CpuState) = unimplemented(s, Instructions.bvfl)
	override fun bvtl(s: CpuState) = unimplemented(s, Instructions.bvtl)

	// Vectorial utilities

	enum class VectorSize(val id: Int) { Single(1), Pair(2), Triple(3), Quad(4) }
	enum class MatrixSize(val id: Int) { M_2x2(2), M_3x3(3), M_4x4(4);

		companion object {
			val items = arrayListOf(M_2x2, M_2x2, M_2x2, M_3x3, M_4x4)
		}
	}

	// VFPU
	fun matrixRegs(matrixReg: Int, N: MatrixSize, callback: (col: Int, row: Int, r: Int) -> Unit): IntArray {
		val mtx = (matrixReg ushr 2) and 7
		val col = matrixReg and 3

		var row = 0
		var side = 0

		when (N) {
			MatrixSize.M_2x2 -> {
				row = (matrixReg ushr 5) and 2
				side = 2
			}
			MatrixSize.M_3x3 -> {
				row = (matrixReg ushr 6) and 1
				side = 3
			}
			MatrixSize.M_4x4 -> {
				row = (matrixReg ushr 5) and 2
				side = 4
			}
		}

		val transpose = ((matrixReg ushr 5) and 1) != 0

		val regs = IntArray(side * side)
		for (i in 0 until side) {
			for (j in 0 until side) {
				var r = mtx * 4
				if (transpose) {
					r += ((row + i) and 3) + ((col + j) and 3) * 32
				} else {
					r += ((col + j) and 3) + ((row + i) and 3) * 32
				}
				callback(j, i, r)
			}
		}
		return regs
	}

	fun matrixRegsVD(ir: Int, callback: (col: Int, row: Int, r: Int) -> Unit) = matrixRegs(ir.vd, MatrixSize.items[ir.one_two], callback)

	class MatrixContext {
		var side: Int = 0
		var col: Int = 0
		var row: Int = 0
		val ms = FloatArray2(4, 4)
		val md = FloatArray2(4, 4)
		val mt = FloatArray2(4, 4)
	}

	private val mc = MatrixContext()

	fun CpuState.setMatrixVD(callback: MatrixContext.() -> Float) {
		matrixRegsVD(IR) { col, row, r ->
			mc.col = col
			mc.row = row
			setVfpr(r, callback(mc))
		}
	}

	fun CpuState.setMatrixVD_VS(callback: MatrixContext.() -> Float) {
		println("setMatrixVD_VS")
	}

	fun CpuState.setMatrixVD_VSVT(callback: MatrixContext.() -> Float) {
		println("setMatrixVD_VSVT")
	}

	//fun CpuState.setMatrix(leftList: IntArray, generator: (column: Int, row: Int, index: Int) -> Float) {
	//	val side = sqrt(leftList.size.toDouble()).toInt()
	//	var n = 0
	//	for (i in 0 until side) {
	//		for (j in 0 until side) {
	//			setVfpr(leftList[n++], generator(j, i, n))
	//		}
	//	}
	//}

	// @TODO: Precalculate this! & mark as inline once this is simplified!
	fun vectorRegisters(vectorReg: Int, N: VectorSize, callback: (index: Int, r: Int) -> Unit) {
		val mtx = (vectorReg ushr 2) and 0b111
		val col = vectorReg and 0b11
		val row: Int
		val length: Int
		var transpose = ((vectorReg ushr 5) and 0b1) != 0

		when (N) {
			VectorSize.Single -> {
				transpose = false
				row = (vectorReg ushr 5) and 3
				length = 1
			}
			VectorSize.Pair -> {
				row = (vectorReg ushr 5) and 2
				length = 2
			}
			VectorSize.Triple -> {
				row = (vectorReg ushr 6) and 1
				length = 3
			}
			VectorSize.Quad -> {
				row = (vectorReg ushr 5) and 2
				length = 4
			}
		}

		for (i in 0 until length) {
			callback(i, mtx * 4 + if (transpose) {
				((row + i) and 3) + col * 32
			} else {
				col + ((row + i) and 3) * 32
			})
		}
	}

	class VfpuContext {
		val tempVS = FloatArray(16)
		val tempVD = FloatArray(16)
		val tempVT = FloatArray(16)

		var vectorSize: Int = 0
		val vs = tempVS
		val vd = tempVD
		val vt = tempVT
	}

	val vfpuContext = VfpuContext()

	fun readVS() {
	}

	fun writeVD() {
	}

	fun CpuState.setVD_VS(destSize: Int = IR.one_two, srcSize: Int = IR.one_two, callback: VfpuContext.(i: Int) -> Float) {
		println("setVD_VS")
	}

	fun CpuState.setVD_VSVT(destSize: Int = IR.one_two, srcSize: Int = IR.one_two, callback: VfpuContext.(i: Int) -> Float) {
		println("setVD_VSVT")
	}

	fun transformValues(input: FloatArray, output: FloatArray, info: Int, enabled: Boolean) {
		if (!enabled) {
			for (n in 0 until input.size) output[n] = input[n]
		} else {
			for (n in 0 until input.size) {
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
						else -> invalidOp
					}
				} else {
					if (sourceAbsolute) input[sourceIndex].absoluteValue else input[sourceIndex]
				}

				output[n] = if (sourceNegate) -value else value
			}
		}
	}

	//fun getMatrixRegs(matrixReg: Int, N: MatrixSize) {
	//	var mtx = (matrixReg ushr 2) and 7
	//	var col = matrixReg and 3
//
	//	var row = 0;
	//	var side = 0;
//
	//	when (N) {
	//		MatrixSize.M_2x2 -> {row = (matrixReg ushr 5) and 2; side = 2 }
	//		MatrixSize.M_3x3 -> {row = (matrixReg ushr 6) and 1; side = 3 }
	//		MatrixSize.M_4x4 -> {row = (matrixReg ushr 5) and 2; side = 4 }
	//		else -> invalidOp
	//	}
//
	//	var transpose = (matrixReg >> 5) & 1;
//
	//	var regs: number[] = new Array(side * side);
	//	for (var i = 0; i < side; i++) {
	//		for (var j = 0; j < side; j++) {
	//		var index = mtx * 4;
	//		if (transpose) {
	//			index += ((row + i) & 3) + ((col + j) & 3) * 32;
	//		} else {
	//			index += ((col + j) & 3) + ((row + i) & 3) * 32;
	//		}
	//		regs[j * side + i] = index;
	//	}
	//	}
	//	return regs;
	//}

	//fun getMatrixRegsVD(i: Int) = getMatrixRegs(i.vd, i.one_two)

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

		constructor(value: Double) : this(value.toFloat())

		companion object {
			val values = values()
			operator fun get(index: Int) = values[index]
		}
	}
}

