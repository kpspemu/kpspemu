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

class Dynarek2X64Gen : X64Builder() {
	val args = arrayOf(X64Reg64.RCX, X64Reg64.RDX, X64Reg64.R8, X64Reg64.R9)

	fun generate(func: D2Func): ByteArray {
		func.body.generate()
		return getBytes()
	}

	fun D2Stm.generate(): Unit = when (this) {
		is D2Stm.Stms -> {
			for (child in children) child.generate()
		}
		is D2Stm.Return -> {
			expr.generate()
			popRAX()
			retn()
		}
		is D2Stm.Write -> {
			popRAX()
		}
		else -> TODO("$this")
	}

	fun D2Expr<*>.generate(): Unit {
		when (this) {
			is D2Expr.ILit -> {
				movEax(this.lit)
				push(X64Reg64.RAX)
			}
			is D2Expr.FLit -> TODO("$this")
			is D2Expr.IBinop -> {
				l.generate()
				r.generate()
				popRAX()
				popRBX()
				when (this.op) {
					D2Binop.ADD -> add(X64Reg32.EAX, X64Reg32.EBX)
					D2Binop.SUB -> sub(X64Reg32.EAX, X64Reg32.EBX)
					D2Binop.MUL -> mul(X64Reg32.EBX)
					D2Binop.DIV -> TODO("$this")
					D2Binop.REM -> TODO("$this")
				}
				pushRAX()
			}
			is D2Expr.Ref -> TODO("$this")
			else -> TODO("$this")
		}
	}
}