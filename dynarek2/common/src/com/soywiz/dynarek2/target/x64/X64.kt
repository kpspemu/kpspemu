package com.soywiz.dynarek2.target.x64

import com.soywiz.dynarek2.*

// typealias D2KFunc = (regs: D2Memory?, mem: D2Memory?, temps: D2Memory?, external: Any?) -> Int
// RAX: <result>
// RBX: <temp>
/////////////////////////
// RCX: regs
// RDX: mem
// R8: temps
// R9: external
/////////////////////////
class Dynarek2X64Gen : X64Builder(), Dynarek2Generator {
	val args = arrayOf(X64Reg64.RCX, X64Reg64.RDX, X64Reg64.R8, X64Reg64.R9)

	override fun generate(func: D2Func): D2Result {
		func.body.generate()
		val mem = NewD2Memory(getBytes())
		return D2Result(Unit, mem) { mem.free() }
	}

	fun D2Stm.generate(): Unit = when (this) {
		is D2Stm.Return -> {
			expr.generate()
			popRAX()
			retn()
		}
		is D2Stm.Write -> {
			popRAX()
		}
		else -> TODO()
	}

	fun D2Expr<*>.generate(): Unit {
		when (this) {
			is D2Expr.ILit -> {
				movEax(this.lit)
				push(X64Reg64.RAX)
			}
			is D2Expr.FLit -> TODO()
			is D2Expr.Binop -> {
				when (type) {
					D2INT -> {
						l.generate()
						r.generate()
						popRAX()
						popRBX()
						when (this.op) {
							D2Binop.ADD -> add(X64Reg32.EAX, X64Reg32.EBX)
							D2Binop.SUB -> sub(X64Reg32.EAX, X64Reg32.EBX)
							D2Binop.MUL -> mul(X64Reg32.EBX)
							D2Binop.DIV -> TODO()
							D2Binop.REM -> TODO()
						}
						pushRAX()
					}
					D2FLOAT -> TODO()
				}
			}
			is D2Expr.Unop -> TODO()
			is D2Expr.Invoke -> TODO()
			is D2Expr.Read -> TODO()
			else -> TODO()
		}
	}
}