package com.soywiz.kpspemu.cpu.dynarec

import com.soywiz.dynarek.*
import com.soywiz.kpspemu.cpu.CpuState
import com.soywiz.kpspemu.cpu.InstructionDispatcher
import com.soywiz.kpspemu.cpu.InstructionEvaluator

class DynarekMethodBuilder : BaseDynarecMethodBuilder() {
	override fun add(i: Int, s: InstructionInfo) = s { RD = RS + RT }
	override fun addiu(i: Int, s: InstructionInfo) = s { RT = RS + S_IMM16 }
	override fun sll(i: Int, s: InstructionInfo) = s { RD = RT shl POS }
	override fun lui(i: Int, s: InstructionInfo) = s { RT = (U_IMM16_V shl 16).lit }
}

data class InstructionInfo(var PC: Int, var IR: Int)

open class BaseDynarecMethodBuilder : InstructionEvaluator<InstructionInfo>() {
	val stms = StmBuilder(Unit::class, CpuState::class, Unit::class)
	val dispatcher = InstructionDispatcher(this)

	fun generateFunction() = DFunction1(DVOID, DClass(CpuState::class), stms.build())

	private val ii = InstructionInfo(0, 0)
	fun dispatch(pc: Int, i: Int) {
		ii.PC = pc
		ii.IR = i
		return dispatcher.dispatch(ii, pc, i)
	}

	fun StmBuilder<Unit, CpuState, Unit>.getRegister(n: Int): DExpr<Int> {
		return when (n) {
			0 -> 0.lit
			else -> p0[CpuState.getGprProp(n)]
		}
	}

	fun StmBuilder<Unit, CpuState, Unit>.setRegister(n: Int, value: DExpr<Int>) {
		if (n != 0) SET(p0[CpuState.getGprProp(n)], value)
	}

	var InstructionInfo.RD: DExpr<Int>
		set(value) = stms.run { setRegister(IR.rd, value) }
		get() = stms.run { getRegister(IR.rd) }

	var InstructionInfo.RS: DExpr<Int>
		set(value) = stms.run { setRegister(IR.rs, value) }
		get() = stms.run { getRegister(IR.rs) }

	var InstructionInfo.RT: DExpr<Int>
		set(value) = stms.run { setRegister(IR.rt, value) }
		get() = stms.run { getRegister(IR.rt) }

	val InstructionInfo.POS: DExpr<Int> get() = stms.run { IR.pos.lit }
	val InstructionInfo.S_IMM16: DExpr<Int> get() = stms.run { IR.s_imm16.lit }
	val InstructionInfo.U_IMM16: DExpr<Int> get() = stms.run { IR.u_imm16.lit }

	val InstructionInfo.S_IMM16_V: Int get() = stms.run { IR.s_imm16 }
	val InstructionInfo.U_IMM16_V: Int get() = stms.run { IR.u_imm16 }

	val Int.lit: DExpr<Int> get() = DLiteral(this)

	operator fun DExpr<Int>.plus(that: DExpr<Int>) = DBinopInt(this, IBinop.ADD, that)
	infix fun DExpr<Int>.shl(that: DExpr<Int>) = DBinopInt(this, IBinop.SHL, that)

	operator inline fun InstructionInfo.invoke(callback: InstructionInfo.() -> Unit) = callback(this)
}
