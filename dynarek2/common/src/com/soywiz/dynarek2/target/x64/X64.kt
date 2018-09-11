package com.soywiz.dynarek2.target.x64

import com.soywiz.dynarek2.*

// typealias D2KFunc = (regs: D2Memory?, mem: D2Memory?, temps: D2Memory?, external: Any?) -> Int
// RAX: <result>
/////////////////////////

class X64ABI(
	val windows: Boolean,
	val args: Array<Reg64>,
	val xmmArgs: Array<RegXmm>,
	val preserved: Array<Reg64>,
	val xmmPreserved: Array<RegXmm>
) {
	companion object {
		// Microsoft x64 ABI (Windows)
		// https://en.wikipedia.org/wiki/X86_calling_conventions#Microsoft_x64_calling_convention
		val Microsoft = X64ABI(
			windows = true,
			args = arrayOf(Reg64.RCX, Reg64.RDX, Reg64.R8, Reg64.R9),
			xmmArgs = arrayOf(RegXmm.XMM0, RegXmm.XMM1, RegXmm.XMM2, RegXmm.XMM3),
			preserved = arrayOf(Reg64.RBX, Reg64.RBP, Reg64.RDI, Reg64.RSI, Reg64.RSP),
			xmmPreserved = arrayOf(RegXmm.XMM6, RegXmm.XMM7)
		)

		// System V AMD64 ABI (Solaris, Linux, FreeBSD, macOS)
		// https://en.wikipedia.org/wiki/X86_calling_conventions#System_V_AMD64_ABI
		val SystemV = X64ABI(
			windows = false,
			args = arrayOf(Reg64.RDI, Reg64.RSI, Reg64.RDX, Reg64.RCX),
			xmmArgs = arrayOf(RegXmm.XMM0, RegXmm.XMM1, RegXmm.XMM2, RegXmm.XMM3),
			preserved = arrayOf(Reg64.RBP, Reg64.RBX),
			xmmPreserved = arrayOf()
		)
	}
}


class Dynarek2X64Gen(
	val context: D2Context,
	val name: String?,
	val debug: Boolean,
	val abi: X64ABI = if (isNativeWindows) X64ABI.Microsoft else X64ABI.SystemV
) : X64Builder() {
	val REGS_IDX = 0
	val MEM_IDX = 1
	val TEMPS_IDX = 2
	val EXT_IDX = 3

	val ABI_ARGS get() = abi.args
	val PRESERVED_ARGS get() = abi.preserved

	val REGS_ARG = ABI_ARGS[0]
	val MEM_ARG = ABI_ARGS[1]
	val TEMP_ARG = ABI_ARGS[2]
	val EXT_ARG = ABI_ARGS[3]

	var allocatedFrame = 0

	var frameStackPos = 0

	fun pushRegisterInFrame(reg: Reg64): Int {
		writeMem(reg, Reg64.RBP, -8 - (frameStackPos * 8))
		frameStackPos++
		if (frameStackPos >= allocatedFrame) error("FRAME TOO BIG")
		return frameStackPos - 1
	}

	fun pushRegisterInFrame(reg: RegXmm): Int {
		writeMem(reg, Reg64.RBP, -8 - (frameStackPos * 8))
		frameStackPos++
		if (frameStackPos >= allocatedFrame) error("FRAME TOO BIG")
		return frameStackPos - 1
	}

	fun popRegisterInFrame(reg: Reg64) {
		frameStackPos--
		readMem(reg, Reg64.RBP, -8 - (frameStackPos * 8))
	}

	fun popRegisterInFrame(reg: RegXmm) {
		frameStackPos--
		readMem(reg, Reg64.RBP, -8 - (frameStackPos * 8))
	}

	fun getFrameItem(reg: Reg64, index: Int) {
		readMem(reg, Reg64.RBP, -8 - (index * 8))
	}

	fun getFrameItem(reg: RegXmm, index: Int) {
		readMem(reg, Reg64.RBP, -8 - (index * 8))
	}

	fun setFrameItem(reg: Reg64, index: Int) {
		writeMem(reg, Reg64.RBP, -8 - (index * 8))
	}

	fun setFrameItem(reg: RegXmm, index: Int) {
		writeMem(reg, Reg64.RBP, -8 - (index * 8))
	}

	fun restoreArgs() {
		for ((index, reg) in ABI_ARGS.withIndex()) getFrameItem(reg, index)
	}

	fun restoreArg(reg: Reg64, index: Int) {
		getFrameItem(reg, index)
	}

	// @TODO: Support different ABIs
	private var rbxIndex = -1
	private var rsiIndex = -1
	private var rdiIndex = -1

	private fun prefix() {
		push(Reg64.RBP)
		mov(Reg64.RBP, Reg64.RSP)
		sub(Reg64.RSP, 512)
		allocatedFrame = 512

		for (reg in ABI_ARGS) pushRegisterInFrame(reg)

		// This should support both supported ABIs
		rbxIndex = pushRegisterInFrame(Reg64.RBX)
		if (abi.windows) {
			rsiIndex = pushRegisterInFrame(Reg64.RSI)
			rdiIndex = pushRegisterInFrame(Reg64.RDI)
		}
	}

	private fun doReturn() {
		// This should support both supported ABIs
		restoreArg(Reg64.RBX, rbxIndex)
		if (abi.windows) {
			restoreArg(Reg64.RSI, rsiIndex)
			restoreArg(Reg64.RDI, rdiIndex)
		}

		mov(Reg64.RSP, Reg64.RBP)
		pop(Reg64.RBP)
		retn()
	}

	fun generateDummy(): ByteArray {
		mov(Reg32.EAX, 123456)
		retn()
		return getBytes()
	}

	fun generate(func: D2Func): ByteArray {
		prefix()
		func.body.generate()
		doReturn() // GUARD in the case a return is missing
		return getBytes()
	}

	fun D2Expr.Ref<*>.loadAddress(): Int {
		offset.generate()
		popRegisterInFrame(Reg64.RDX)
		movEax(size.bytes)
		mul(Reg32.EDX)
		restoreArg(Reg64.RDX, memSlot.index)
		add(Reg64.RAX, Reg64.RDX)
		return pushRegisterInFrame(Reg64.RAX)
	}

	fun D2Stm.generate(): Unit {
		subframe {
			when (this) {
				is D2Stm.Stms -> {
					for (child in children) child.generate()
				}
				is D2Stm.BExpr -> {
					subframe {
						expr.generate()
						popRegisterInFrame(Reg64.RAX)
						if (this is D2Stm.Return) {
							doReturn()
						}
					}
				}
				is D2Stm.Set<*> -> {
					val memSlot = ref.memSlot
					val offset = ref.offset
					val size = ref.size

					val foffset = if (offset is D2Expr.ILit) {
						val valueIndex = value.generate()
						getFrameItem(Reg64.RDX, valueIndex)
						restoreArg(Reg64.RDI, memSlot.index)
						offset.lit * size.bytes
					} else {
						val offsetIndex = ref.loadAddress()
						val valueIndex = value.generate()
						restoreArg(Reg64.RDI, offsetIndex)
						restoreArg(Reg64.RDX, valueIndex)
						0
					}
					when (size) {
						D2Size.BYTE -> writeMem(Reg8.DL, Reg64.RDI, foffset)
						D2Size.SHORT -> writeMem(Reg16.DX, Reg64.RDI, foffset)
						D2Size.INT, D2Size.FLOAT -> writeMem(Reg32.EDX, Reg64.RDI, foffset)
						D2Size.LONG -> TODO("$size")
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
			}
		}
	}

	//companion object {
	//	val SHL_NAME = D2FuncName("shl")
	//	val SHR_NAME = D2FuncName("shr")
	//	val USHR_NAME = D2FuncName("ushr")
	//}

	fun D2Expr<*>.generate(): Int {
		when (this) {
			is D2Expr.ILit -> {
                mov(Reg32.EAX, this.lit)
            }
			is D2Expr.FLit -> {
				mov(Reg32.EAX, this.lit.toRawBits())
			}
			is D2Expr.IBinOp -> {
				generate(l to Reg64.RAX, r to Reg64.RCX)

				when (this.op) {
					D2IBinOp.ADD -> add(Reg32.EAX, Reg32.ECX)
					D2IBinOp.SUB -> sub(Reg32.EAX, Reg32.ECX)
					D2IBinOp.MUL -> imul(Reg32.ECX)
					D2IBinOp.DIV -> idiv(Reg32.ECX)
					D2IBinOp.REM -> {
						idiv(Reg32.ECX)
						mov(Reg64.RAX, Reg64.RDX)
					}
					D2IBinOp.SHL -> shl(Reg32.EAX, Reg8.CL)
					D2IBinOp.SHR -> shr(Reg32.EAX, Reg8.CL)
					D2IBinOp.USHR -> ushr(Reg32.EAX, Reg8.CL)
					D2IBinOp.AND -> and(Reg32.EAX, Reg32.ECX)
					D2IBinOp.OR -> or(Reg32.EAX, Reg32.ECX)
					D2IBinOp.XOR -> xor(Reg32.EAX, Reg32.ECX)
				}
			}
			is D2Expr.FBinOp -> {
				generateXmm(l to RegXmm.XMM0, r to RegXmm.XMM1)
				when (this.op) {
					D2FBinOp.ADD -> addss(RegXmm.XMM0, RegXmm.XMM1)
					D2FBinOp.SUB -> subss(RegXmm.XMM0, RegXmm.XMM1)
					D2FBinOp.MUL -> mulss(RegXmm.XMM0, RegXmm.XMM1)
					D2FBinOp.DIV -> divss(RegXmm.XMM0, RegXmm.XMM1)
					//D2FBinOp.REM -> TODO("frem")
				}
				return pushRegisterInFrame(RegXmm.XMM0)
			}
			is D2Expr.IComOp -> {
				val label1 = Label()
				val label2 = Label()

				generateJump2(this, label1, isTrue = true)
				mov(Reg32.EAX, 0)
				jmp(label2)
				place(label1)
				mov(Reg32.EAX, 1)
				place(label2)
			}
			is D2Expr.Ref-> {
				val foffset = if (offset is D2Expr.ILit) {
					restoreArg(Reg64.RDX, memSlot.index)
					offset.lit * size.bytes
				} else {
					this.loadAddress()
					popRegisterInFrame(Reg64.RDX)
					0
				}
				// @TODO: MOVZX (mov zero extension) http://faydoc.tripod.com/cpu/movzx.htm
 				when (size) {
					D2Size.BYTE -> {
						mov(Reg32.EAX, 0)
						readMem(Reg8.AL, Reg64.RDX, foffset)
					}
					D2Size.SHORT -> {
						mov(Reg32.EAX, 0)
						readMem(Reg16.AX, Reg64.RDX, foffset)
					}
					D2Size.INT, D2Size.FLOAT -> readMem(Reg32.EAX, Reg64.RDX, foffset)
					D2Size.LONG -> TODO("LONG")
				}
			}
			is D2Expr.Invoke<*> -> {
				val func = context.getFunc(func)
				subframe {
					val frameIndices = args.map { it.generate() }
					val param = ParamsReader(abi)
					for (n in 0 until args.size) {
						val frameIndex = frameIndices[n]
						val type = func.args.getOrElse(0) { D2INT }
						if (type == D2FLOAT) {
							getFrameItem(param.getXmm(), frameIndex)
						} else {
							getFrameItem(param.getInt(), frameIndex)
						}
					}
				}
				callAbsolute(func.address)
				return if (func.rettype == D2FLOAT) {
					pushRegisterInFrame(RegXmm.XMM0)
				} else {
					pushRegisterInFrame(Reg64.RAX)
				}
			}
			is D2Expr.External -> {
				restoreArg(Reg64.RAX, EXT_IDX)
			}
			else -> TODO("$this")
		}
		return pushRegisterInFrame(Reg64.RAX)
	}

	class ParamsReader(val abi: X64ABI) {
		var ipos = 0
		var fpos = 0
		fun reset() {
			ipos = 0
			fpos = 0
		}
		fun getInt(): Reg64 = abi.args[ipos++]
		fun getXmm(): RegXmm = abi.xmmArgs[ipos++]
	}

	fun generate(vararg items: Pair<D2Expr<*>, Reg64>) {
		subframe {
			val indices = items.map { (expr, _) ->
				when {
					expr is D2Expr.ILit -> -1
					else -> expr.generate()
				}
			}
			for ((it, index) in items.zip(indices)) {
				val expr = it.first
				val reg = it.second
				when {
					expr is D2Expr.ILit -> mov(reg.to32(), expr.lit)
					else -> getFrameItem(reg, index)
				}
			}
		}
	}

	fun generateXmm(vararg items: Pair<D2Expr<*>, RegXmm>) {
		subframe {
			val indices = items.map { (expr, _) ->
				expr.generate()
			}
			for ((it, index) in items.zip(indices)) {
				val expr = it.first
				val reg = it.second
				getFrameItem(reg, index)
			}
		}
	}

	fun generateJumpComp(l: D2ExprI, op: D2CompOp, r: D2ExprI, label: Label, isTrue: Boolean) {
		generate(l to Reg64.RBX, r to Reg64.RCX)

		cmp(Reg32.EBX, Reg32.ECX)
		when (if (isTrue) op else op.negated) {
			D2CompOp.EQ -> je(label)
			D2CompOp.NE -> jne(label)
			D2CompOp.LT -> jl(label)
			D2CompOp.LE -> jle(label)
			D2CompOp.GT -> jg(label)
			D2CompOp.GE -> jge(label)
		}
	}

	fun generateJump2(e: D2Expr.IComOp, label: Label, isTrue: Boolean) {
		generateJumpComp(e.l, e.op, e.r, label, isTrue)
	}

	fun generateJump(e: D2Expr<*>, label: Label, isTrue: Boolean) {
		when (e) {
			is D2Expr.IComOp -> generateJump2(e, label, isTrue)
			else -> {
				e.generate()
				popRegisterInFrame(Reg64.RAX)
				cmp(Reg32.EAX, 0)
				if (isTrue) jne(label) else je(label)
			}
		}
	}

	fun generateJumpFalse(e: D2Expr<*>, label: Label): Unit = generateJump(e, label, false)
	fun generateJumpTrue(e: D2Expr<*>, label: Label): Unit = generateJump(e, label, true)
	fun generateJumpAlways(label: Label): Unit = jmp(label)

	inline fun subframe(callback: () -> Unit) {
		val current = this.frameStackPos
		try {
			callback()
		} finally {
			this.frameStackPos = current
		}
	}
}

inline class RegContent(val type: Int) {
	companion object {
		fun temp(index: Int) = RegContent(0x0000 + index)
		fun arg(index: Int) = RegContent(0x1000 + index)
	}
}
