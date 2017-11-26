package com.soywiz.kpspemu.cpu.dis

import com.soywiz.korio.lang.Console
import com.soywiz.korio.lang.format
import com.soywiz.korio.util.hex
import com.soywiz.kpspemu.cpu.InstructionDecoder
import com.soywiz.kpspemu.cpu.InstructionOpcodeDecoder
import com.soywiz.kpspemu.cpu.InstructionType
import com.soywiz.kpspemu.mem.Memory

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
			//"%j" -> "0x%08X".format((pc and 0xF0000000.toInt()) or i.jump_address)
				"%j" -> "0x%08X".format((pc and (-268435456)) or i.jump_address)
				"%J" -> gprStr(i.rs)

				"%d" -> gprStr(i.rd)
				"%s" -> gprStr(i.rs)
				"%t" -> gprStr(i.rt)

				"%D" -> fprStr(i.fd)
				"%S" -> fprStr(i.fs)
				"%T" -> fprStr(i.ft)

				"%a" -> "${i.pos}"
				"%O" -> "PC + ${i.s_imm16}"
				"%C" -> "0x%04X".format(i.syscall)
				"%I" -> "${i.u_imm16}"
				"%i" -> "${i.s_imm16}"
				else -> type
			}
		}

		return "${op.name} $params"
	}
}

fun Memory.disasm(pc: Int): String = Disassembler.disasm(pc, this.lw(pc))
fun Memory.disasmMacro(pc: Int): String = try {
	Disassembler.disasmMacro(pc, this.lw(pc))
} catch (e: IndexOutOfBoundsException) {
	"invalid(PC=0x%08X)".format(pc)
}

fun Memory.getPrintInstructionAt(address: Int): String = address.hex + " : " + disasmMacro(address)
fun Memory.printInstructionAt(address: Int) = Console.error(getPrintInstructionAt(address))


