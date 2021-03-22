package com.soywiz.kpspemu.cpu.dis

import com.soywiz.klogger.*
import com.soywiz.korio.lang.*
import com.soywiz.kpspemu.cpu.*
import com.soywiz.kpspemu.mem.*
import com.soywiz.krypto.encoding.*

interface NameProvider {
    fun getName(addr: Int): String?
}

object DummyNameProvider : NameProvider {
    override fun getName(addr: Int): String? = null
}

object Disassembler : InstructionDecoder() {
    private val PERCENT_REGEX = Regex("%\\w+")

    fun gprStr(i: Int) = "r$i"
    fun fprStr(i: Int) = "f$i"

    fun disasmMacro(pc: Int, i: Int, nameProvider: NameProvider = DummyNameProvider): String {
        if (i == 0) return "nop"
        val op = InstructionOpcodeDecoder(i)
        return disasm(op, pc, i, nameProvider)
    }

    fun disasm(pc: Int, i: Int, nameProvider: NameProvider = DummyNameProvider): String =
        disasm(InstructionOpcodeDecoder(i), pc, i, nameProvider)

    fun disasm(op: InstructionType, pc: Int, i: Int, nameProvider: NameProvider = DummyNameProvider): String {
        var comments = ""
        val params = op.format.replace(PERCENT_REGEX) {
            val type = it.groupValues[0]
            when (type) {
            //"%j" -> "0x%08X".format((pc and 0xF0000000.toInt()) or i.jump_address)
                "%j" -> {
                    val aaddr = (pc and (-268435456)) or i.jump_address
                    val aname = nameProvider.getName(aaddr)
                    if (aname != null) comments += aname
                    "0x%08X".format(aaddr)
                }
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
                "%c" -> "0x%04X".format(i.syscall)
                "%I" -> "${i.u_imm16}"
                "%i" -> "${i.s_imm16}"
                else -> type
            }
        }

        var out = "${op.name} $params"
        if (comments.isNotEmpty()) out += " ; $comments"
        return out
    }
}

fun Memory.disasm(pc: Int, nameProvider: NameProvider = DummyNameProvider): String =
    Disassembler.disasm(pc, this.lw(pc), nameProvider)

fun Memory.disasmMacro(pc: Int, nameProvider: NameProvider = DummyNameProvider): String = try {
    Disassembler.disasmMacro(pc, this.lw(pc), nameProvider)
} catch (e: IndexOutOfBoundsException) {
    "invalid(PC=0x%08X)".format(pc)
} catch (e: Exception) {
    "error: ${e.message}"
}

fun Memory.getPrintInstructionAt(address: Int, nameProvider: NameProvider = DummyNameProvider): String =
    "${address.hex} : ${disasmMacro(address, nameProvider)}"

fun Memory.printInstructionAt(address: Int, nameProvider: NameProvider = DummyNameProvider) =
    Console.error(getPrintInstructionAt(address, nameProvider))


