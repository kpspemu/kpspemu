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

class Dynarek2X64Gen(val context: D2Context, val name: String?, val debug: Boolean) : X64Builder() {
	// @TODO: Does windows has a different ABI for x64?
	//val ABI_ARGS = arrayOf(X64Reg64.RCX, X64Reg64.RDX, X64Reg64.R8, X64Reg64.R9)
	val ABI_ARGS = arrayOf(Reg64.RDI, Reg64.RSI, Reg64.RDX, Reg64.RCX)

	private fun prefix() {
		push(Reg64.RBP)
		mov(Reg64.RBP, Reg64.RSP)

		push(Reg64.RBX)
		push(Reg64.RSI)
		push(Reg64.RDI)

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
		pop(Reg64.RDI)
		pop(Reg64.RSI)
		pop(Reg64.RBX)

		pop(Reg64.RBP)
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
			expr.generate(Reg64.RAX)
			doReturn()
		}
		is D2Stm.Write -> {
			val target = Reg64.RAX
			val baseReg = ABI_ARGS[ref.memSlot.index]
			value.generate(target)
			if (ref.offset is D2Expr.ILit) {
				writeMem(target.to32(), baseReg, ref.offset.lit * 4)
			} else {
				TODO()
			}
		}
		is D2Stm.If -> {
			if (sfalse == null) {
				// IF
				val endLabel = Label()
				generateJumpFalse(cond, endLabel)
				strue.generate()
				place(endLabel)
			} else {
				// IF+ELSE
				val elseLabel = Label()
				val endLabel = Label()
				generateJumpFalse(cond, elseLabel)
				strue.generate()
				generateJumpAlways(endLabel)
				place(elseLabel)
				sfalse?.generate()
				place(endLabel)
			}
		}
		is D2Stm.While -> {
			val startLabel = Label()
			val endLabel = Label()
			place(startLabel)
			generateJumpFalse(cond, endLabel)
			body.generate()
			generateJumpAlways(startLabel)
			place(endLabel)
		}
		else -> TODO("$this")
	}

	//companion object {
	//	val SHL_NAME = D2FuncName("shl")
	//	val SHR_NAME = D2FuncName("shr")
	//	val USHR_NAME = D2FuncName("ushr")
	//}

	fun D2Expr<*>.generate(target: Reg64): Unit {
		when (this) {
			is D2Expr.ILit -> mov(target.to32(), this.lit)
			is D2Expr.FLit -> TODO("$this")
			is D2Expr.IBinOp -> {
				when (this.op) {
					D2BinOp.SHL, D2BinOp.SHR, D2BinOp.USHR -> {
						pushPop(Reg64.RCX) {
							l.generate(target)
							r.generate(Reg64.RCX)
							when (this.op) {
								D2BinOp.SHL -> shl(target.to32(), Reg8.CL)
								D2BinOp.SHR -> shr(target.to32(), Reg8.CL)
								D2BinOp.USHR -> ushr(target.to32(), Reg8.CL)
								else -> TODO()
							}
						}
					}
					else -> {
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
									pushPopIfRequired(Reg64.RAX, target) {
										pushPopIfRequired(Reg64.RDX, target) {
											if (target != Reg64.RAX) mov(Reg64.RAX, target)
											mul(temp.to32())
											if (target != Reg64.RAX) mov(target, Reg64.RAX)
										}
									}
								}
								D2BinOp.DIV -> TODO("$this")
								D2BinOp.REM -> TODO("$this")
								D2BinOp.SHL -> {
									shl(target.to32(), temp.to32())
								}
								D2BinOp.SHR -> shr(target.to32(), temp.to32())
								D2BinOp.USHR -> ushr(target.to32(), temp.to32())
								D2BinOp.AND -> and(target.to32(), temp.to32())
								D2BinOp.OR -> or(target.to32(), temp.to32())
								D2BinOp.XOR -> xor(target.to32(), temp.to32())
							}
						}
					}
				}
			}
			is D2Expr.IComOp -> {
				val temp = tryGetTempReg(except = target)

				l.generate(target)
				pushPopIfRequired(temp, target) {
					requestPreserve(target) {
						r.generate(temp)
					}
					cmp(target.to32(), temp.to32())
					val label1 = Label()
					val label2 = Label()

					when (this.op) {
						D2CompOp.EQ -> je(label1)
						D2CompOp.NE -> jne(label1)
						D2CompOp.LT -> jl(label1)
						D2CompOp.LE -> jle(label1)
						D2CompOp.GT -> jg(label1)
						D2CompOp.GE -> jge(label1)
					}

					mov(target.to32(), 0)
					jmp(label2)
					place(label1)
					mov(target.to32(), 1)
					place(label2)
				}
			}
			is D2Expr.Ref-> {
				if (size != D2Size.INT) TODO("$size")
				val baseReg = ABI_ARGS[memSlot.index]
				if (offset is D2Expr.ILit) {
					readMem(target.to32(), baseReg, offset.lit * 4)
				} else {
					pushPopIfRequired(Reg64.RAX, target) {
						pushPopIfRequired(Reg64.RBX, target) {
							offset.generate(target)
							popRBX()
							movEax(4)
							mul(Reg32.EBX)
							add(Reg64.RAX, baseReg)
							readMem(target.to32(), Reg64.RAX, 0)
						}
					}
				}
			}
			is D2Expr.InvokeI -> {
				for ((index, arg) in args.withIndex()) {
					arg.generate(ABI_ARGS[index])
				}
				pushPopIfRequired(Reg64.RAX, target) {
					callAbsolute(context.getFunc(func))
					mov(target, Reg64.RAX)
				}
			}
			else -> TODO("$this")
		}
	}

	fun generateJump(e: D2Expr<*>, label: Label, isTrue: Boolean): Unit {
		e.generate(Reg64.RAX)
		cmp(Reg32.EAX, 0)
		if (isTrue) jne(label) else je(label)
	}

	fun generateJumpFalse(e: D2Expr<*>, label: Label): Unit = generateJump(e, label, false)
	fun generateJumpTrue(e: D2Expr<*>, label: Label): Unit = generateJump(e, label, true)
	fun generateJumpAlways(label: Label): Unit = jmp(label)

	val usedRegs = BooleanArray(32)

	private val tempRegs = arrayOf(Reg64.RBX, Reg64.RSI, Reg64.RDI) // @TODO: Must fix high operations with regs
	//private val tempRegs = arrayOf(X64Reg64.RBX, X64Reg64.RAX)

	fun tryGetTempReg(except: Reg64): Reg64 {
		for (r in tempRegs) if (r != except && !usedRegs[r.index]) return r
		for (r in tempRegs) if (r != except) return r
		return Reg64.RBX
	}

	inline fun requestPreserve(reg: Reg64, callback: () -> Unit) {
		val old = usedRegs[reg.index]
		usedRegs[reg.index] = true
		try {
			callback()
		} finally {
			usedRegs[reg.index] = old
		}
	}

	inline fun pushPopIfRequired(reg: Reg64, target: Reg64, callback: () -> Unit) {
		val used = if (reg != target) usedRegs[reg.index] else false
		if (used) push(reg)
		try {
			callback()
		} finally {
			if (used) pop(reg)
		}
	}

	inline fun pushPop(reg: Reg64, callback: () -> Unit) {
		push(reg)
		try {
			callback()
		} finally {
			pop(reg)
		}
	}
}
