package com.soywiz.kpspemu.cpu.dis

import com.soywiz.korio.lang.format
import com.soywiz.kpspemu.cpu.InstructionDecoder
import com.soywiz.kpspemu.cpu.InstructionOpcodeDecoder
import com.soywiz.kpspemu.cpu.InstructionType

object Disassembler : InstructionDecoder() {
	private val PERCENT_REGEX = Regex("%\\w+")

	fun gprStr(i: Int) = "r$i"
	fun fprStr(i: Int) = "f$i"

	fun disasmMacro(pc: Int, i: Int): String {
		if (i == 0) return "nop"
		val op = InstructionOpcodeDecoder(i)
		return disasm(op, pc, i)
	}

	fun disasm(pc: Int, i: Int): String = disasm(InstructionOpcodeDecoder(i), pc, i)

	fun disasm(op: InstructionType, pc: Int, i: Int): String {
		val params = op.format.replace(PERCENT_REGEX) {
			val type = it.groupValues[0]
			when (type) {
				"%d" -> gprStr(i.rd)
				"%s" -> gprStr(i.rs)
				"%a" -> "${i.pos}"
				"%O" -> "PC + ${i.s_imm16}"
				"%t" -> gprStr(i.rt)
				"%C" -> "0x%04X".format(i.syscall)
				"%I" -> "${i.u_imm16}"
				"%i" -> "${i.s_imm16}"
				else -> type
			}
		}

		return "${op.name} $params"
	}
}
