package com.soywiz.kpspemu.cpu.assembler

import com.soywiz.korge.util.*
import com.soywiz.korio.error.*
import com.soywiz.korio.stream.*
import com.soywiz.kpspemu.cpu.*
import kotlin.collections.set

@NativeThreadLocal
object Assembler : InstructionDecoder() {
    val mnemonicGpr get() = CpuState.gprInfosByMnemonic

    fun parseGpr(str: String): Int {
        if (str.startsWith("r")) return str.substring(1).toInt()
        return mnemonicGpr[str]?.index ?: invalidOp("Unknown GPR $str")
    }

    fun String.parseInt(context: Context): Int {
        context.labels[this]?.let { return it }
        val str = this.replace("_", "")
        if (str.startsWith("0x")) return str.substring(2).toInt(16)
        return str.toInt()
    }

    fun assembleSingleInt(str: String, pc: Int = 0, context: Context = Context()): Int =
        assembleSingle(str, pc, context).value

    data class ParsedInstruction(val str: String) {
        val parts = str.split(Regex("\\s+"), limit = 2)
        val nameRaw = parts.getOrElse(0) { "" }
        val name = nameRaw.toLowerCase()
        val args = parts.getOrElse(1) { "" }
        val argsParsed = args.trim().split(',').map { it.trim() }
    }

    fun assembleSingle(str: String, pc: Int = 0, context: Context = Context()): InstructionData {
        val pi = ParsedInstruction(str)
        val instruction = Instructions.instructionsByName[pi.name] ?: invalidOp("Unknown instruction '${pi.name}'")
        val out = InstructionData(instruction.vm.value, pc)
        val replacements = instruction.replacements
        val formatRegex = instruction.formatRegex
        val match = formatRegex.matchEntire(pi.args)
                ?: invalidOp("${instruction.format} -> $formatRegex doesn't match ${pi.args}")
        val results = replacements.zip(match.groupValues.drop(1))
        //println(context.labels)
        for ((pat, value) in results) {
            when (pat) {
                "%s" -> out.rs = parseGpr(value)
                "%d" -> out.rd = parseGpr(value)
                "%t" -> out.rt = parseGpr(value)
                "%a" -> out.pos = value.parseInt(context)
                "%i" -> out.s_imm16 = value.parseInt(context)
                "%I" -> out.u_imm16 = value.parseInt(context)
                "%O" -> out.s_imm16 = value.parseInt(context) - pc - 4
                "%j" -> out.jump_address = value.parseInt(context)
                else -> invalidOp("Unsupported $pat :: value=$value in instruction $str")
            }
        }
        return out
    }

    fun assembleTo(sstr: String, out: SyncStream, pc: Int = out.position.toInt(), context: Context = Context()): Int {
        val labelParts = sstr.split(":", limit = 2).reversed()
        val str = labelParts[0].trim()
        val label = labelParts.getOrNull(1)?.trim()
        if (label != null) {
            context.labels[label] = pc
            //println("$label -> ${pc.hex}")
        }
        //println("${pc.hex}: $str")

        val pi = ParsedInstruction(str)
        val start = out.position
        when (pi.name) {
            "" -> {
                // None
            }
            "nop" -> {
                out.write32_le(assembleSingleInt("sll zero, zero, 0", pc + 0, context))
            }
            "li" -> {
                val reg = pi.argsParsed[0]
                val value = pi.argsParsed[1].parseInt(context)
                if ((value ushr 16) != 0) {
                    out.write32_le(assembleSingleInt("lui $reg, ${value ushr 16}", pc + 0, context))
                    out.write32_le(assembleSingleInt("ori $reg, $reg, ${value and 0xFFFF}", pc + 4, context))
                } else {
                    out.write32_le(assembleSingleInt("ori $reg, r0, ${value and 0xFFFF}", pc + 0, context))
                }
            }
            else -> {
                out.write32_le(assembleSingleInt(str, pc, context))
            }
        }
        val end = out.position
        val written = (end - start).toInt()
        //println("   : $written (${pi.name})")
        return written
    }

    class Context {
        val labels = LinkedHashMap<String, Int>()
    }

    fun assembleTo(lines: List<String>, out: SyncStream, pc: Int = out.position.toInt(), context: Context = Context()) {
        var cpc = pc
        for (line in lines) {
            if (line.isEmpty()) continue
            cpc += assembleTo(line, out, cpc, context)
        }
    }

    fun assemble(lines: List<String>, pc: Int = 0, context: Context = Context()): ByteArray {
        return MemorySyncStreamToByteArray {
            assembleTo(lines, this, pc, context)
        }
    }

    fun assemble(vararg lines: String, pc: Int = 0, context: Context = Context()): ByteArray =
        assemble(lines.toList(), pc = pc, context = context)
}