package com.soywiz.kpspemu.cpu.interpreter

import com.soywiz.kmem.*
import com.soywiz.korim.color.RGBA
import com.soywiz.korio.error.invalidOp
import com.soywiz.korio.lang.Console
import com.soywiz.korio.lang.format
import com.soywiz.korio.util.*
import com.soywiz.korma.math.Math
import com.soywiz.korma.math.isAlmostZero
import com.soywiz.korma.math.reinterpretAsFloat
import com.soywiz.korma.math.reinterpretAsInt
import com.soywiz.kpspemu.cpu.*
import com.soywiz.kpspemu.cpu.CpuState.VCondition
import com.soywiz.kpspemu.cpu.dis.NameProvider
import com.soywiz.kpspemu.cpu.dis.disasmMacro
import com.soywiz.kpspemu.hle.manager._thread
import com.soywiz.kpspemu.mem.Memory
import com.soywiz.kpspemu.util.*
import kotlin.math.*

class CpuInterpreter(var cpu: CpuState, val breakpoints: Breakpoints, val nameProvider: NameProvider, var trace: Boolean = false) {
	val dispatcher = InstructionDispatcher(InstructionInterpreter)

	fun steps(count: Int, trace: Boolean = false): Int {
		val mem = cpu.mem.getFastMem()
		//val mem = null
		return if (mem != null) {
			stepsFastMem(mem, cpu.mem.getFastMemOffset(Memory.MAIN_OFFSET) - Memory.MAIN_OFFSET, count, trace)
		} else {
			stepsNormal(count, trace)
		}
	}

	fun stepsNormal(count: Int, trace: Boolean): Int {
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
				if (trace) doTrace(sPC, cpu)
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
			checkException(sPC, e)
		} finally {
			cpu.totalExecuted += n
		}
		return n
	}

	fun stepsFastMem(mem: FastMemory, memOffset: Int, count: Int, trace: Boolean): Int {
		val i32 = mem.i32
		val cpu = this.cpu
		var n = 0
		var sPC = 0
		val breakpointsEnabled = breakpoints.enabled
		try {
			while (n < count) {
				sPC = cpu._PC and 0x0FFFFFFF
				if (trace) doTrace(sPC, cpu)
				if (breakpointsEnabled && breakpoints[sPC]) throw BreakpointException(cpu, sPC)
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

	private val VDEST2 = IntArray2(4, 4)
	private val VDEST = IntArray(16)
	private val VSRC = IntArray(16)
	private val VTARGET = IntArray(16)

	private val VSRCF = FloatArray(16)

	private fun _lv_x(s: CpuState, size: Int) = s {
		getVectorRegisters(VSRC, IR.vt5_1, VectorSize(size))
		val start = RS_IMM14
		for (n in 0 until size) s.VFPRI[VSRC[n]] = mem.lw(start + n * 4)
	}

	private fun _sv_x(s: CpuState, size: Int) = s {
		getVectorRegisters(VSRC, IR.vt5_1, VectorSize(size))
		val start = RS_IMM14
		for (n in 0 until size) mem.sw(start + n * 4, s.VFPRI[VSRC[n]])
	}

	override fun lv_s(s: CpuState) = _lv_x(s, 1)
	override fun lv_q(s: CpuState) = _lv_x(s, 4)

	override fun sv_s(s: CpuState) = _sv_x(s, 1)
	override fun sv_q(s: CpuState) = _sv_x(s, 4)

	override fun lvl_q(s: CpuState) = s {
		getVectorRegisters(VSRC, IR.vt5_1, VectorSize.Quad)
		mem.lvl_q(RS_IMM14) { i, value -> s.setVfprI(VSRC[i], value) }
	}

	override fun lvr_q(s: CpuState) = s {
		getVectorRegisters(VSRC, IR.vt5_1, VectorSize.Quad)
		mem.lvr_q(RS_IMM14) { i, value -> s.setVfprI(VSRC[i], value) }
	}

	override fun svl_q(s: CpuState) = s {
		getVectorRegisters(VSRC, IR.vt5_1, VectorSize.Quad)
		mem.svl_q(RS_IMM14) { getVfprI(VSRC[it]) }
	}

	override fun svr_q(s: CpuState) = s {
		getVectorRegisters(VSRC, IR.vt5_1, VectorSize.Quad)
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

	override fun vt4444_q(s: CpuState) = s._vtXXXX_q(this::cc_8888_to_4444)
	override fun vt5551_q(s: CpuState) = s._vtXXXX_q(this::cc_8888_to_5551)
	override fun vt5650_q(s: CpuState) = s._vtXXXX_q(this::cc_8888_to_5650)

	private fun _vc2i(s: CpuState, func: (index: Int, value: Int) -> Int) = s {
		setVDI_VS(destSize = 4, srcSize = 1) { func(it, vsi.x) }
	}

	override fun vc2i(s: CpuState) = _vc2i(s) { index, value -> (value shl ((3 - index) * 8)) and 0xFF000000.toInt() }
	override fun vuc2i(s: CpuState) = _vc2i(s) { index, value -> ((((value ushr (index * 8)) and 0xFF) * 0x01010101) shr 1) and 0x80000000.toInt().inv() }

	private fun _vs2i(s: CpuState, func: (index: Int, value: Int) -> Int) = s {
		val size = IR.one_two
		getVectorRegisters(VSRC, IR.vs, VectorSize(size))
		getVectorRegisters(VDEST, IR.vd, VectorSize(size * 2))
		for (n in 0 until size) {
			val value = VFPRI[VSRC[n]]
			VFPRI[VDEST[n * 2 + 0]] = func(0, value)
			VFPRI[VDEST[n * 2 + 1]] = func(1, value)
		}
	}

	override fun vs2i(s: CpuState) = _vs2i(s) { index, value -> value.extract(index * 16, 16) shl 16 }
	override fun vus2i(s: CpuState) = _vs2i(s) { index, value -> value.extract(index * 16, 16) shl 15 }

	private fun _vi2c(s: CpuState, gen: (value: Int) -> Int) = s {
		getVectorRegisterValuesInt(VSRC, IR.vs, VectorSize.Quad)
		getVectorRegisters(VDEST, IR.vd, VectorSize.Single)
		VFPRI[VDEST[0]] = RGBA.packFast(gen(VSRC[0]), gen(VSRC[1]), gen(VSRC[2]), gen(VSRC[3]))
	}

	override fun vi2c(s: CpuState) = _vi2c(s) { it.extract8(24) }
	override fun vi2uc(s: CpuState) = _vi2c(s) { if (it < 0) 0 else it.extract8(23) }

	private fun _vi2s(s: CpuState, gen: (value: Int) -> Int) = s {
		val size = IR.one_two
		val dsize = size / 2
		getVectorRegisterValuesInt(VSRC, IR.vs, VectorSize(size))
		getVectorRegisters(VDEST, IR.vd, VectorSize(dsize))
		for (n in 0 until dsize) {
			val l = gen(VSRC[n * 2 + 0])
			val r = gen(VSRC[n * 2 + 1])
			VFPRI[VDEST[n]] = l or (r shl 16)
		}
	}

	override fun vi2s(s: CpuState) = _vi2s(s) { it ushr 16 }
	override fun vi2us(s: CpuState) = _vi2s(s) { if (it < 0) 0 else it shr 15 }


	override fun vi2f(s: CpuState) = s {
		val size = IR.one_two
		getVectorRegisterValuesInt(VSRC, IR.vs, VectorSize(size))
		getVectorRegisters(VDEST, IR.vd, VectorSize(size))
		for (n in 0 until size) {
			VFPR[VDEST[n]] = VSRC[n] * 2f.pow(-IR.imm5)
		}
	}

	private fun _vf2ix(s: CpuState, func: (value: Float, imm5: Int) -> Int) = s {
		val size = IR.one_two
		getVectorRegisterValuesFloat(VSRCF, IR.vs, VectorSize(size))
		getVectorRegisters(VDEST, IR.vd, VectorSize(size))
		for (n in 0 until size) {
			//println("${VSRCF[n]} : ${IR.imm5}")
			val value = VSRCF[n]
			VFPRI[VDEST[n]] = if (value.isNaN()) 0x7FFFFFFF else func(value, IR.imm5)
		}
	}

	// @TODO: Verify these ones!
	override fun vf2id(s: CpuState) = _vf2ix(s) { value, imm5 -> floor(value * 2f.pow(imm5)).toInt() }

	override fun vf2iu(s: CpuState) = _vf2ix(s) { value, imm5 -> ceil(value * 2f.pow(imm5)).toInt() }
	override fun vf2in(s: CpuState) = _vf2ix(s) { value, imm5 -> Math.rint((value * 2f.pow(imm5))) }
	override fun vf2iz(s: CpuState) = _vf2ix(s) { value, imm5 -> val rs = value * 2f.pow(imm5); if (value >= 0) floor(rs).toInt() else ceil(rs).toInt() }

	override fun vf2h(s: CpuState) = s {
		val size = IR.one_two
		val dsize = size / 2
		getVectorRegisterValuesInt(VSRC, IR.vs, VectorSize(size))
		getVectorRegisters(VDEST, IR.vd, VectorSize(dsize))
		for (n in 0 until dsize) {
			val l = HalfFloat.floatBitsToHalfFloatBits(VSRC[n * 2 + 0])
			val r = HalfFloat.floatBitsToHalfFloatBits(VSRC[n * 2 + 1])
			VFPRI[VDEST[n]] = (l) or (r shl 16)
		}
	}

	override fun vh2f(s: CpuState) = s {
		val size = IR.one_two
		val dsize = size * 2
		getVectorRegisterValuesInt(VSRC, IR.vs, VectorSize(size))
		getVectorRegisters(VDEST, IR.vd, VectorSize(dsize))
		for (n in 0 until size) {
			val value = VSRC[n]
			VFPRI[VDEST[n * 2 + 0]] = HalfFloat.halfFloatBitsToFloatBits(value.extract(0, 16))
			VFPRI[VDEST[n * 2 + 1]] = HalfFloat.halfFloatBitsToFloatBits(value.extract(16, 16))
		}
	}

	// Move this outside
	fun HalfFloat.toFloat() = this.f
	fun Float.toHalfFloat() = HalfFloat(this)
	data class HalfFloat(val v: Char) {
		constructor(v: Float) : this(floatBitsToHalfFloatBits(v.reinterpretAsInt()).toChar())
		val f: Float get() = halfFloatBitsToFloatBits(v.toInt()).reinterpretAsFloat()
		override fun toString(): String = "$f"
		fun toBits() = v

		companion object {
			fun fromBits(v: Char) = HalfFloat(v)

			fun halfFloatBitsToFloat(imm16: Int): Float = Float.fromBits(halfFloatBitsToFloatBits(imm16))
			fun floatToHalfFloatBits(i: Float): Int = floatBitsToHalfFloatBits(i.toRawBits())

			fun halfFloatBitsToFloatBits(imm16: Int): Int {
				val s = imm16 shr 15 and 0x00000001 // sign
				var e = imm16 shr 10 and 0x0000001f // exponent
				var f = imm16 shr 0 and 0x000003ff // fraction

				// need to handle 0x7C00 INF and 0xFC00 -INF?
				if (e == 0) {
					// need to handle +-0 case f==0 or f=0x8000?
					if (f == 0) {
						// Plus or minus zero
						return s shl 31
					}
					// Denormalized number -- renormalize it
					while (f and 0x00000400 == 0) {
						f = f shl 1
						e -= 1
					}
					e += 1
					f = f and 0x00000400.inv()
				} else if (e == 31) {
					return if (f == 0) {
						// Inf
						s shl 31 or 0x7f800000
					} else s shl 31 or 0x7f800000 or f
					// NaN
					// fraction is not shifted by PSP
				}

				e += (127 - 15)
				f = f shl 13

				return s shl 31 or (e shl 23) or f
			}

			fun floatBitsToHalfFloatBits(i: Int): Int {
				val s = i shr 16 and 0x00008000              // sign
				val e = (i shr 23 and 0x000000ff) - (127 - 15) // exponent
				var f = i shr 0 and 0x007fffff              // fraction

				// need to handle NaNs and Inf?
				if (e <= 0) {
					if (e < -10) {
						return if (s != 0) {
							// handle -0.0
							0x8000
						} else 0
					}
					f = f or 0x00800000 shr 1 - e
					return s or (f shr 13)
				} else if (e == 0xff - (127 - 15)) {
					if (f == 0) {
						// Inf
						return s or 0x7c00
					}
					// NAN
					f = f shr 13
					f = 0x3ff // PSP always encodes NaN with this value
					return s or 0x7c00 or f or if (f == 0) 1 else 0
				}
				return if (e > 30) {
					// Overflow
					s or 0x7c00
				} else s or (e shl 10) or (f shr 13)
			}
		}
	}

	override fun viim(s: CpuState) = s { VT = S_IMM16.toFloat() }
	override fun vcst(s: CpuState) = s { VD = VfpuConstants[IR.imm5].value }
	override fun mtv(s: CpuState) = s { VD_I = RT }
	override fun vpfxt(s: CpuState) = s { vpfxt.setEnable(IR) }
	override fun vpfxd(s: CpuState) = s { vpfxd.setEnable(IR) }
	override fun vpfxs(s: CpuState) = s { vpfxs.setEnable(IR) }
	override fun vavg(s: CpuState) =  s {
		val size = IR.one_two
		getVectorRegisterValuesFloat(VSRCF, IR.vs, VectorSize(size))
		getVectorRegisters(VDEST, IR.vd, VectorSize.Single)
		VFPR[VDEST[0]] = ((0 until size).sumByDouble { (VSRCF[it] / size).toDouble() }).toFloat()
	}
	override fun vfad(s: CpuState) = s {
		val size = IR.one_two
		getVectorRegisterValuesFloat(VSRCF, IR.vs, VectorSize(size))
		getVectorRegisters(VDEST, IR.vd, VectorSize.Single)
		VFPR[VDEST[0]] = ((0 until size).sumByDouble { VSRCF[it].toDouble() }).toFloat()
	}
	override fun vrot(s: CpuState) = s {
		val vectorSize = IR.one_two
		val imm5 = IR.imm5
		val cosIndex = imm5.extract(0, 2)
		val sinIndex = imm5.extract(2, 2)
		val negateSin = imm5.extractBool(4)

		setVD_VS(vectorSize, 1) {
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
	override fun vzero(s: CpuState) = s { setVD_ { 0f } }
	override fun vone(s: CpuState) = s { setVD_ { 1f } }

	// Vector operations (one operand)
	override fun vmov(s: CpuState) = s { setVD_VS { vs[it] } }
	override fun vabs(s: CpuState) = s { setVD_VS { abs(vs[it]) } }
	override fun vsqrt(s: CpuState) = s { setVD_VS { sqrt(vs[it]) } }
	override fun vneg(s: CpuState) = s { setVD_VS { -vs[it] } }
	override fun vsat0(s: CpuState) = s { setVD_VS { vs[it].clampf(0f, 1f) } }
	override fun vsat1(s: CpuState) = s { setVD_VS { vs[it].clampf(-1f, 1f) } }
	override fun vrcp(s: CpuState) = s { setVD_VS { 1f / vs[it] } }
	override fun vrsq(s: CpuState) = s { setVD_VS { 1f / sqrt(vs[it]) } }
	override fun vsin(s: CpuState) = s { setVD_VS { sinv1(vs[it]) } }
	override fun vasin(s: CpuState) = s { setVD_VS { asinv1(vs[it]) } }
	override fun vnsin(s: CpuState) = s { setVD_VS { -sinv1(vs[it]) } }
	override fun vcos(s: CpuState) = s { setVD_VS { cosv1(vs[it]) } }
	override fun vexp2(s: CpuState) = s { setVD_VS { 2f.pow(vs[it]) } }
	override fun vrexp2(s: CpuState) = s { setVD_VS { 1f / 2f.pow(vs[it]) } }
	override fun vlog2(s: CpuState) = s { setVD_VS { log2(vs[it]) } }
	override fun vnrcp(s: CpuState) = s { setVD_VS { -1f / vs[it] } }
	override fun vsgn(s: CpuState) = s { setVD_VS { sign(vs[it]) } }
	override fun vocp(s: CpuState) = s { setVD_VS { 1f - vs[it] } }
	override fun vbfy1(s: CpuState) = s { setVD_VS {
		when (it) {
			0 -> vs.x + vs.y
			1 -> vs.x - vs.y
			2 -> vs.z + vs.w
			3 -> vs.z - vs.w
			else -> invalidOp
		}
	} }
	override fun vbfy2(s: CpuState) = s { setVD_VS {
		when (it) {
			0 -> vs.x + vs.z
			1 -> vs.y + vs.w
			2 -> vs.x - vs.z
			3 -> vs.y - vs.w
			else -> invalidOp
		}
	} }
	override fun vsrt1(s: CpuState) = s { setVD_VS { when (it) {
		0 -> min(vs.x, vs.y)
		1 -> max(vs.x, vs.y)
		2 -> min(vs.z, vs.w)
		3 -> max(vs.z, vs.w)
		else -> invalidOp
	} } }
	override fun vsrt2(s: CpuState) = s { setVD_VS { vs.run { when (it) {
		0 -> min(x, w)
		1 -> min(y, z)
		2 -> max(y, z)
		3 -> max(x, w)
		else -> invalidOp
	} } } }
	override fun vsrt3(s: CpuState) = s { setVD_VS { when (it) {
		0 -> max(vs.x, vs.y)
		1 -> min(vs.x, vs.y)
		2 -> max(vs.z, vs.w)
		3 -> min(vs.z, vs.w)
		else -> invalidOp
	} } }
	override fun vsrt4(s: CpuState) = s { setVD_VS { when (it) {
		0 -> max(vs.x, vs.w)
		1 -> max(vs.y, vs.z)
		2 -> min(vs.y, vs.z)
		3 -> min(vs.x, vs.w)
		else -> invalidOp
	} } }

	// Vector operations (two operands)
	override fun vsge(s: CpuState) = s { setVD_VSVT { if (vs[it] >= vt[it]) 1f else 0f } }
	override fun vslt(s: CpuState) = s { setVD_VSVT { if (vs[it] < vt[it]) 1f else 0f } }
	override fun vscmp(s: CpuState) = s { setVD_VSVT { vs[it].compareTo(vt[it]).toFloat() } }

	override fun vadd(s: CpuState) = s { setVD_VSVT { vs[it] + vt[it] } }
	override fun vsub(s: CpuState) = s { setVD_VSVT { vs[it] - vt[it] } }
	override fun vmul(s: CpuState) = s { setVD_VSVT { vs[it] * vt[it] } }
	override fun vdiv(s: CpuState) = s { setVD_VSVT { vs[it] / vt[it] } }
	override fun vmin(s: CpuState) = s { setVD_VSVT { min(vs[it], vt[it]) } }
	override fun vmax(s: CpuState) = s { setVD_VSVT { max(vs[it], vt[it]) } }
	override fun vcrs_t(s: CpuState) = s { setVD_VSVT {
		when (it) {
			0 -> vs.y * vt.z
			1 -> vs.z * vt.x
			2 -> vs.x * vt.y
			else -> invalidOp
		}
	} }
	override fun vcrsp_t(s: CpuState) = s { setVD_VSVT {
		when (it) {
			0 -> +vs.y * vt.z - vs.z * vt.y
			1 -> +vs.z * vt.x - vs.x * vt.z
			2 -> +vs.x * vt.y - vs.y * vt.x
			else -> invalidOp
		}
	} }
	override fun vqmul(s: CpuState) = s { setVD_VSVT {
		when (it) {
			0 -> +vs.x * vt.w + vs.y * vt.z - vs.z * vt.y + vs.w * vt.x
			1 -> -vs.x * vt.z + vs.y * vt.w + vs.z * vt.x + vs.w * vt.y
			2 -> +vs.x * vt.y - vs.y * vt.x + vs.z * vt.w + vs.w * vt.z
			3 -> -vs.x * vt.x - vs.y * vt.y - vs.z * vt.z + vs.w * vt.w
			else -> invalidOp
		}
	} }
	override fun vdot(s: CpuState) = s { setVD_VSVT(destSize = 1) {
		((0 until vsSize).sumByDouble { (vs[it] * vt[it]).toDouble() }).toFloat()
	} }
	override fun vscl(s: CpuState) = s { setVD_VSVT(targetSize = 1) { vs[it] * vt.x } }

	override fun vhdp(s: CpuState) = s { setVD_VSVT(destSize = 1) {
		vs[vsSize - 1] = 1f
		(0 until vsSize).sumByDouble {  (vs[it] * vt[it]).toDouble() }.toFloat()
	} }
	override fun vdet(s: CpuState) = s { setVD_VSVT(destSize = 1) { vs.x * vt.y - vs.y * vt.x } }
	override fun vcmp(s: CpuState) = s {
		val size = IR.one_two
		var cc = 0
		_VSVT {
			val cond = when (IR.imm4) {
				VCondition.FL -> false
				VCondition.EQ -> vs[it] == vt[it]
				VCondition.LT -> vs[it] < vt[it]
				VCondition.LE -> vs[it] <= vt[it]

				VCondition.TR -> true
				VCondition.NE -> vs[it] != vt[it]
				VCondition.GE -> vs[it] >= vt[it]
				VCondition.GT -> vs[it] > vt[it]

				VCondition.EZ -> (vs[it] == 0f) || (vt[it] == -0f)
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
	}
	private fun _vcmovtf(s: CpuState, truth: Boolean) = s {
		val ccidx = IR.imm3
		setVD_VDVS {
			val cond = when (ccidx) {
				0, 1, 2, 3, 4, 5 -> VFPR_CC(ccidx)
				6 -> VFPR_CC(it)
				7 -> true
				else -> false
			}
			if (cond == truth) vs[it] else vd[it]
		}
	}

	override fun vcmovf(s: CpuState) = _vcmovtf(s, true)
	override fun vcmovt(s: CpuState) = _vcmovtf(s, false)

	override fun vfim(s: CpuState) = s { setVD_(destSize = 1) { HalfFloat.halfFloatBitsToFloat(U_IMM16) } }
	override fun vwbn(s: CpuState) = s { setVD_VS(destSize = 1) { println("vwbn not implemented!"); -777f } }
	override fun vsbn(s: CpuState) = s { setVD_VSVT { scalab(vs[it], vti[it]) } }

	// Matrix operations
	override fun vmzero(s: CpuState) = s { setMatrixVD { 0f } }
	override fun vmone(s: CpuState) = s { setMatrixVD { 1f } }
	override fun vmidt(s: CpuState) = s { setMatrixVD { if (row == col) 1f else 0f } }
	override fun vmmov(s: CpuState) = s { setMatrixVD_VS { ms[col, row] } }
	override fun vmmul(s: CpuState) = s { setMatrixVD_VSVT {
		(0 until side).map { ms[col, row] * mt[row, col] }.sum() }
	}

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
	override fun mfvc(s: CpuState) = s {
		RT = VFPRC[IR.imm7]
	}
	override fun mtvc(s: CpuState) = s {
		vpfxs
		when (IR.imm7) {
			0 -> return@s this.vpfxs.setUnknown()
			1 -> return@s this.vpfxt.setUnknown()
			2 -> return@s this.vpfxd.setUnknown()
		}
		VFPRC[IR.imm7] = RT

	}

	private fun _vtfm_x(s: CpuState, size: Int) = s { vfpuContext.run {
		getVectorRegisterValues(b_vt, IR.vt, VectorSize(size))

		for (n in 0 until size) {
			getVectorRegisterValues(b_vs, IR.vs + n, VectorSize(size))
			vfpuContext.vd[n] = (0 until size).sumByDouble { (vs[it] * vt[it]).toDouble() }.toFloat()
		}

		setVectorRegisterValues(b_vd, IR.vd, VectorSize(size))
	} }

	private fun _vhtfm_x(s: CpuState, size: Int) = s { vfpuContext.run {
		getVectorRegisterValues(b_vt, IR.vt, VectorSize(size - 1))

		vt[size - 1] = 1f
		for (n in 0 until size) {
			getVectorRegisterValues(b_vs, IR.vs + n, VectorSize(size))
			vfpuContext.vd[n] = (0 until size).sumByDouble { (vs[it] * vt[it]).toDouble() }.toFloat()
		}

		setVectorRegisterValues(b_vd, IR.vd, VectorSize(size))
	} }

	override fun vtfm2(s: CpuState) = _vtfm_x(s, 2)
	override fun vtfm3(s: CpuState) = _vtfm_x(s, 3)
	override fun vtfm4(s: CpuState) = _vtfm_x(s, 4)

	override fun vhtfm2(s: CpuState) = _vhtfm_x(s, 2)
	override fun vhtfm3(s: CpuState) = _vhtfm_x(s, 2)
	override fun vhtfm4(s: CpuState) = _vhtfm_x(s, 2)

	override fun vmscl(s: CpuState) = s {
		val scale = vfpuContext.sreadVt(s, size = 1)[0]
		setMatrixVD_VS { ms[col, row] * scale }
	}

	override fun vidt(s: CpuState) = unimplemented(s, Instructions.vidt)
	override fun vnop(s: CpuState) = unimplemented(s, Instructions.vnop)
	override fun vsync(s: CpuState) = unimplemented(s, Instructions.vsync)
	override fun vflush(s: CpuState) = unimplemented(s, Instructions.vflush)
	override fun vrnds(s: CpuState) = unimplemented(s, Instructions.vrnds)
	override fun vrndi(s: CpuState) = unimplemented(s, Instructions.vrndi)
	override fun vrndf1(s: CpuState) = unimplemented(s, Instructions.vrndf1)
	override fun vrndf2(s: CpuState) = unimplemented(s, Instructions.vrndf2)
	override fun vmfvc(s: CpuState) = unimplemented(s, Instructions.vmfvc)
	override fun vmtvc(s: CpuState) = unimplemented(s, Instructions.vmtvc)
	override fun mfvme(s: CpuState) = unimplemented(s, Instructions.mfvme)
	override fun mtvme(s: CpuState) = unimplemented(s, Instructions.mtvme)
	override fun vlgb(s: CpuState) = unimplemented(s, Instructions.vlgb)
	override fun vsbz(s: CpuState) = unimplemented(s, Instructions.vsbz)
	override fun vsocp(s: CpuState) = unimplemented(s, Instructions.vsocp)
	override fun bvf(s: CpuState) = unimplemented(s, Instructions.bvf)
	override fun bvt(s: CpuState) = unimplemented(s, Instructions.bvt)
	override fun bvfl(s: CpuState) = unimplemented(s, Instructions.bvfl)
	override fun bvtl(s: CpuState) = unimplemented(s, Instructions.bvtl)

	// Vectorial utilities

	enum class VectorSize(val id: Int) {
		Single(1), Pair(2), Triple(3), Quad(4);

		companion object {
			val items = arrayOf(Single, Single, Pair, Triple, Quad)
			operator fun invoke(size: Int) = items[size]
		}
	}

	enum class MatrixSize(val id: Int) { M_2x2(2), M_3x3(3), M_4x4(4);
		companion object {
			val items = arrayOf(M_2x2, M_2x2, M_2x2, M_3x3, M_4x4)
			operator fun invoke(size: Int) = items[size]
		}
	}

	fun CpuState.getMatrixRegsValues(out: FloatArray2, matrixReg: Int, N: MatrixSize): Int {
		val side = getMatrixRegs(tempRegs2, matrixReg, N)
		for (j in 0 until side) for (i in 0 until side) out[j, i] = getVfpr(tempRegs2[j, i])
		return side
	}

	fun getMatrixRegs(out: IntArray2, matrixReg: Int, N: MatrixSize): Int {
		val side = N.id
		val mtx = (matrixReg ushr 2) and 7
		val col = matrixReg and 3
		val transpose = ((matrixReg ushr 5) and 1) != 0
		val row = when (N) {
			MatrixSize.M_2x2 -> (matrixReg ushr 5) and 2
			MatrixSize.M_3x3 -> (matrixReg ushr 6) and 1
			MatrixSize.M_4x4 -> (matrixReg ushr 5) and 2
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
		getMatrixRegs(VDEST2, IR.vd, MatrixSize(side))
		mc.side = side
		for (col in 0 until side) for (row in 0 until side) {
			setVfpr(VDEST2[col, row], callback(mc.setPos(col, row)))
		}
	}

	fun CpuState.setMatrixVD_VS(side: Int = IR.one_two, callback: MatrixContext.() -> Float) {
		getMatrixRegs(VDEST2, IR.vd, MatrixSize(side))
		getMatrixRegsValues(mc.ms, IR.vs, MatrixSize(side))
		mc.side = side
		for (col in 0 until side) for (row in 0 until side) {
			setVfpr(VDEST2[col, row], callback(mc.setPos(col, row)))
		}
	}

	fun CpuState.setMatrixVD_VSVT(side: Int = IR.one_two, callback: MatrixContext.() -> Float) {
		getMatrixRegs(VDEST2, IR.vd, MatrixSize(side))
		getMatrixRegsValues(mc.ms, IR.vs, MatrixSize(side))
		getMatrixRegsValues(mc.mt, IR.vt, MatrixSize(side))

		mc.side = side
		for (col in 0 until side) for (row in 0 until side) {
			setVfpr(VDEST2[col, row], callback(mc.setPos(col, row)))
		}
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

	private val tempRegs = IntArray(16)
	private val tempRegs2 = IntArray2(4, 4)

	fun getVectorRegister(vectorReg: Int, N: VectorSize = VectorSize.Single, index: Int = 0): Int {
		vectorRegisters(vectorReg, N) { i, r -> tempRegs[i] = r }
		return tempRegs[index]
	}

	fun getVectorRegisters(out: IntArray, vectorReg: Int, N: VectorSize) {
		vectorRegisters(vectorReg, N) { i, r -> out[i] = r }
	}

	fun CpuState.getVectorRegisterValuesInt(out: IntArray, vectorReg: Int, N: VectorSize) {
		vectorRegisters(vectorReg, N) { i, r -> out[i] = getVfprI(r) }
	}

	fun CpuState.setVectorRegisterValuesInt(inp: IntArray, vectorReg: Int, N: VectorSize) {
		vectorRegisters(vectorReg, N) { i, r -> setVfprI(r, inp[i]) }
	}

	fun CpuState.getVectorRegisterValuesFloat(out: FloatArray, vectorReg: Int, N: VectorSize) {
		vectorRegisters(vectorReg, N) { i, r -> out[i] = getVfpr(r) }
	}

	fun CpuState.setVectorRegisterValuesFloat(inp: FloatArray, vectorReg: Int, N: VectorSize) {
		vectorRegisters(vectorReg, N) { i, r -> setVfpr(r, inp[i]) }
	}

	fun CpuState.getVectorRegisterValues(out: FloatIntBuffer, vectorReg: Int, N: VectorSize) {
		vectorRegisters(vectorReg, N) { i, r -> out.i[i] = getVfprI(r) }
	}

	fun CpuState.setVectorRegisterValues(inp: FloatIntBuffer, vectorReg: Int, N: VectorSize) {
		vectorRegisters(vectorReg, N) { i, r -> setVfprI(r, inp.i[i]) }
	}

	// @TODO: Precalculate this! & mark as inline once this is simplified!
	fun vectorRegisters(vectorReg: Int, N: VectorSize, callback: (index: Int, r: Int) -> Unit) {
		val mtx = vectorReg.extract(2, 3)
		val col = vectorReg.extract(0, 2)
		val row: Int
		val length: Int = N.id
		val transpose = (N != VectorSize.Single) && vectorReg.extractBool(5)

		when (N) {
			VectorSize.Single -> row = (vectorReg ushr 5) and 3
			VectorSize.Pair -> row = (vectorReg ushr 5) and 2
			VectorSize.Triple -> row = (vectorReg ushr 6) and 1
			VectorSize.Quad -> row = (vectorReg ushr 5) and 2
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

		fun CpuState.readVs(reg: Int = IR.vs, size: Int = IR.one_two): Float32Buffer {
			vsSize = size
			getVectorRegisterValues(b_vs, reg, VectorSize(size))
			return vs
		}

		fun CpuState.readVt(reg: Int = IR.vt, size: Int = IR.one_two): Float32Buffer {
			vtSize = size
			getVectorRegisterValues(b_vt, reg, VectorSize(size))
			return vt
		}

		fun CpuState.readVd(reg: Int = IR.vd, size: Int = IR.one_two): Float32Buffer {
			vdSize = size
			getVectorRegisterValues(b_vd, reg, VectorSize(size))
			return vd
		}

		fun CpuState.writeVd(reg: Int = IR.vd, size: Int = IR.one_two) {
			setVectorRegisterValues(b_vd, reg, VectorSize(size))
		}

		fun sreadVs(s: CpuState, reg: Int = s.IR.vs, size: Int = s.IR.one_two) = s.readVs(reg, size)
		fun sreadVt(s: CpuState, reg: Int = s.IR.vt, size: Int = s.IR.one_two) = s.readVt(reg, size)
		fun sreadVd(s: CpuState, reg: Int = s.IR.vd, size: Int = s.IR.one_two) = s.readVd(reg, size)
		fun swriteVd(s: CpuState, reg: Int = s.IR.vd, size: Int = s.IR.one_two) = s.writeVd(reg, size)
	}

	val vfpuContext = VfpuContext()

	fun CpuState.setVD_(destSize: Int = IR.one_two, callback: VfpuContext.(i: Int) -> Float) = vfpuContext.run {
		vdSize = destSize
		for (n in 0 until destSize) vd[n] = callback(vfpuContext, n)
		writeVd(size = destSize)
	}

	fun CpuState.setVD_VS(destSize: Int = IR.one_two, srcSize: Int = IR.one_two, callback: VfpuContext.(i: Int) -> Float) = vfpuContext.run {
		vdSize = destSize
		readVs(size = srcSize)
		for (n in 0 until destSize) vd[n] = callback(vfpuContext, n)
		writeVd(size = destSize)
	}

	fun CpuState.setVDI_VS(destSize: Int = IR.one_two, srcSize: Int = IR.one_two, callback: VfpuContext.(i: Int) -> Int) = vfpuContext.run {
		vdSize = destSize
		readVs(size = srcSize)
		for (n in 0 until destSize) vdi[n] = callback(vfpuContext, n)
		writeVd(size = destSize)
	}

	fun CpuState.setVD_VDVS(destSize: Int = IR.one_two, srcSize: Int = IR.one_two, callback: VfpuContext.(i: Int) -> Float) = vfpuContext.run {
		vdSize = destSize
		readVs(size = srcSize)
		readVd(size = destSize)
		for (n in 0 until destSize) vd[n] = callback(vfpuContext, n)
		writeVd(size = destSize)
	}

	fun CpuState.setVD_VSVT(destSize: Int = IR.one_two, srcSize: Int = IR.one_two, targetSize: Int = srcSize, callback: VfpuContext.(i: Int) -> Float) = vfpuContext.run {
		vfpuContext.vdSize = destSize
		readVs(size = srcSize)
		readVt(size = targetSize)
		for (n in 0 until destSize) vd[n] = callback(vfpuContext, n)
		writeVd(size = destSize)
	}

	fun CpuState._VSVT(destSize: Int = IR.one_two, srcSize: Int = IR.one_two, targetSize: Int = srcSize, callback: VfpuContext.(i: Int) -> Unit) = vfpuContext.run {
		readVs(size = srcSize)
		readVt(size = targetSize)
		for (n in 0 until destSize) callback(vfpuContext, n)
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

		constructor(value: Double) : this(value.toFloat())

		companion object {
			val values = values()
			operator fun get(index: Int) = values[index]
		}
	}
}

