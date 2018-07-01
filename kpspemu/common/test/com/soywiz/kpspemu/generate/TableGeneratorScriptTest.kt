package com.soywiz.kpspemu.generate

import com.soywiz.kmem.*
import com.soywiz.korio.util.*
import com.soywiz.kpspemu.*
import com.soywiz.kpspemu.cpu.*
import kotlin.test.*

fun main(args: Array<String>) {
    val switch = TableGenerator.createSwitch(Instructions.instructions)
    println(switch)
}

class TableGeneratorScriptTest : BaseTest() {
    @Test
    @Ignore
    fun name() {
        val switch = TableGenerator.createSwitch(Instructions.instructions)
        println(switch)
    }

    @Test
    @Ignore
    fun name2() {
        TableGenerator.dumpImpl(Instructions.instructions)
    }

    @Test
    @Ignore
    fun name3() {
        TableGenerator.dumpInstructionList(Instructions.instructions)
    }
}

class TableGenerator {
    private var lastId = 0

    private fun getCommonMask(instructions: List<InstructionType>, baseMask: Int = 0xFFFFFFFF.toInt()): Int {
        var mask = baseMask
        for (i in instructions) {
            mask = mask and i.vm.mask
        }
        //return instructions.reduce { left, item -> left and item.vm.mask, baseMask }
        return mask
    }

    companion object {
        fun createSwitch(instructions: List<InstructionType>): String {
            val writer = Indenter()
            val decodingTable = TableGenerator()
            decodingTable._createSwitch(writer, instructions)
            return writer.toString()
        }

        fun dumpImpl(instructions: List<InstructionType>): Unit {
            for (i in instructions) {
                //println("open fun ${i.name.kescape()}(s: T): Unit = unimplemented(s, ${i.name.quote()})")
                println("open fun ${i.name.kescape()}(s: T): Unit = unimplemented(s, InstructionTable.${i.name.kescape()})")
            }
        }

        fun dumpInstructionList(instructions: List<InstructionType>): Unit {
            for (i in instructions) {
                val addressTypeStr = when (i.addressType) {
                    ADDR_TYPE_NONE -> "ADDR_TYPE_NONE"
                    ADDR_TYPE_REG -> "ADDR_TYPE_REG"
                    ADDR_TYPE_16 -> "ADDR_TYPE_16"
                    ADDR_TYPE_26 -> "ADDR_TYPE_26"
                    else -> TODO()
                }

                val itypes = arrayListOf<String>()
                if ((i.instructionType and INSTR_TYPE_PSP) != 0) itypes += "INSTR_TYPE_PSP"
                if ((i.instructionType and INSTR_TYPE_SYSCALL) != 0) itypes += "INSTR_TYPE_SYSCALL"
                if ((i.instructionType and INSTR_TYPE_B) != 0) itypes += "INSTR_TYPE_B"
                if ((i.instructionType and INSTR_TYPE_LIKELY) != 0) itypes += "INSTR_TYPE_LIKELY"
                if ((i.instructionType and INSTR_TYPE_JAL) != 0) itypes += "INSTR_TYPE_JAL"
                if ((i.instructionType and INSTR_TYPE_JUMP) != 0) itypes += "INSTR_TYPE_JUMP"
                if ((i.instructionType and INSTR_TYPE_BREAK) != 0) itypes += "INSTR_TYPE_BREAK"
                if (itypes.isEmpty()) itypes += "0"

                println(
                    "val ${i.name.kescape()} = ID(${i.name.quote()}, VM(${i.vm.format.quote()}), ${i.format.quote()}, $addressTypeStr, ${itypes.joinToString(
                        " or "
                    )})"
                )
            }

            println("val instructions = listOf(${instructions.map { it.name.kescape() }.joinToString(", ")})")
        }
    }

    private fun _createSwitch(
        writer: Indenter,
        instructions: List<InstructionType>,
        baseMask: Int = 0xFFFFFFFF.toInt(),
        level: Int = 0
    ) {
        if (level >= 10) throw Exception("ERROR: Recursive detection")
        val commonMask = this.getCommonMask(instructions, baseMask)
        val groups = LinkedHashMap<Int, ArrayList<InstructionType>>()
        for (item in instructions) {
            val commonValue = item.vm.value and commonMask
            val group = groups.getOrPut(commonValue) { arrayListOf() }
            group.add(item)
        }

        fun Int.str() = "$this"
        //fun Int.str() = this.hex

        val maskShift = commonMask.countTrailingZeros()
        val shiftedCommonMask = commonMask ushr maskShift

        writer.line("""when (((i shr ${maskShift.str()}) and ${shiftedCommonMask.str()})) {""")
        writer.indent {
            for ((groupKey, group) in groups.toList().sortedBy { it.first ushr maskShift }) {
                //for ((groupKey, group) in groups) {
                val case = "${(groupKey ushr maskShift).str()} ->"
                if (group.size == 1) {
                    writer.line("$case return e.${group[0].name.kescape()}(i, s)")
                } else {
                    writer.line("$case ")
                    writer.indent { this._createSwitch(writer, group, commonMask.inv(), level + 1) }
                }
            }
            writer.line("""else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (${this.lastId++}) failed mask 0x%08X".format(i, pc, ${commonMask.str()}))""")
        }
        writer.line("}")
    }
}