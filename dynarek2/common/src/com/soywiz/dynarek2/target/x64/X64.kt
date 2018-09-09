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
	//val args = arrayOf(X64Reg64.RCX, X64Reg64.RDX, X64Reg64.R8, X64Reg64.R9)
	val args = arrayOf(X64Reg64.RDI, X64Reg64.RSI, X64Reg64.RDX, X64Reg64.RCX)

	private fun prefix() {
		push(X64Reg64.RBP)
		mov(X64Reg64.RBP, X64Reg64.RSP)

		push(X64Reg64.RBX)
		push(X64Reg64.RSI)
		push(X64Reg64.RDI)

		// NonVolatile
		//usedRegs[X64Reg64.RBX.index] = true
		//usedRegs[X64Reg64.RSI.index] = true
		//usedRegs[X64Reg64.RDI.index] = true
		//usedRegs[X64Reg64.R12.index] = true
		//usedRegs[X64Reg64.R13.index] = true
		//usedRegs[X64Reg64.R14.index] = true
		//usedRegs[X64Reg64.R15.index] = true
	}

	private fun doReturn() {
		pop(X64Reg64.RDI)
		pop(X64Reg64.RSI)
		pop(X64Reg64.RBX)

		pop(X64Reg64.RBP)
		retn()
	}

	fun generate(func: D2Func): ByteArray {
		prefix()
		func.body.generate()
		doReturn() // GUARD in the case a return is missing
		return getBytes()
	}

	fun D2Stm.generate(): Unit = when (this) {
		is D2Stm.Stms -> {
			for (child in children) child.generate()
		}
		is D2Stm.Return -> {
			expr.generate(X64Reg64.RAX)
			doReturn()
		}
		is D2Stm.Write -> {
			popRAX()
		}
		else -> TODO("$this")
	}

	fun D2Expr<*>.generate(target: X64Reg64): Unit {
		when (this) {
			is D2Expr.ILit -> mov(target.to32(), this.lit)
			is D2Expr.FLit -> TODO("$this")
			is D2Expr.IBinOp -> {
				val temp = tryGetTempReg(except = target)

				l.generate(target)
				pushPopIfRequired(temp, target) {
					requestPreserve(target) {
						r.generate(temp)
					}
					when (this.op) {
						D2BinOp.ADD -> add(target.to32(), temp.to32())
						D2BinOp.SUB -> sub(target.to32(), temp.to32())
						D2BinOp.MUL -> {
							pushPopIfRequired(X64Reg64.RAX, target) {
								pushPopIfRequired(X64Reg64.RDX, target) {
									if (target != X64Reg64.RAX) mov(X64Reg64.RAX, target)
									mul(temp.to32())
									if (target != X64Reg64.RAX) mov(target, X64Reg64.RAX)
								}
							}
						}
						D2BinOp.DIV -> TODO("$this")
						D2BinOp.REM -> TODO("$this")
					}
				}
			}
			is D2Expr.Ref-> {
				if (size != D2Size.INT) TODO("$size")
				val baseReg = args[memSlot.index]
				if (offset is D2Expr.ILit) {
					readMem(target.to32(), baseReg, offset.lit * 4)
				} else {
					pushPopIfRequired(X64Reg64.RAX, target) {
						pushPopIfRequired(X64Reg64.RBX, target) {
							offset.generate(target)
							popRBX()
							movEax(4)
							mul(X64Reg32.EBX)
							add(X64Reg64.RAX, baseReg)
							readMem(target.to32(), X64Reg64.RAX, 0)
						}
					}
				}
			}
			else -> TODO("$this")
		}
	}

	val usedRegs = BooleanArray(32)

	private val tempRegs = arrayOf(X64Reg64.RBX, X64Reg64.RSI, X64Reg64.RDI) // @TODO: Must fix high operations with regs
	//private val tempRegs = arrayOf(X64Reg64.RBX, X64Reg64.RAX)

	fun tryGetTempReg(except: X64Reg64): X64Reg64 {
		for (r in tempRegs) if (r != except && !usedRegs[r.index]) return r
		for (r in tempRegs) if (r != except) return r
		return X64Reg64.RBX
	}

	inline fun requestPreserve(reg: X64Reg64, callback: () -> Unit) {
		val old = usedRegs[reg.index]
		usedRegs[reg.index] = true
		try {
			callback()
		} finally {
			usedRegs[reg.index] = old
		}
	}

	inline fun pushPopIfRequired(reg: X64Reg64, target: X64Reg64, callback: () -> Unit) {
		val used = if (reg != target) usedRegs[reg.index] else false
		if (used) push(reg)
		callback()
		if (used) pop(reg)
	}
}