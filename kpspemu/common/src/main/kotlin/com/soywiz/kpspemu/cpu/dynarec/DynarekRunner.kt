package com.soywiz.kpspemu.cpu.dynarec

import com.soywiz.dynarek.*
import com.soywiz.dynarek.js.*
import com.soywiz.kds.*
import com.soywiz.korio.util.*
import com.soywiz.kpspemu.cpu.*
import com.soywiz.kpspemu.cpu.dis.*
import com.soywiz.kpspemu.cpu.interpreter.*
import com.soywiz.kpspemu.mem.*

class DynarekRunner(
    var cpu: CpuState,
    val breakpoints: Breakpoints,
    val nameProvider: NameProvider,
    var trace: Boolean = false
) {
    val gcpu = cpu.globalCpuState
    val mcache = gcpu.mcache
    val dispatcher = InstructionDispatcher(InstructionInterpreter(cpu))

    val disasmPcs = IntSet()
    fun steps(count: Int, trace: Boolean = false): Int {
        val dump = false
        //val dump = true
        //val trace = true
        var n = 0
        do {
            if (trace) {
                cpu.dump()
                println("####### ${cpu.PC.hex}:")
            }
            val func = mcache.getFunction(cpu.PC)

            if (dump) {
                if (!disasmPcs.contains(func.pc)) {
                    cpu.dump()
                    println("####### ${cpu.PC.hex}:")
                    disasmPcs.add(func.pc)
                    //if (cpu.PC == 0x089000BC) runBlocking { localCurrentDirVfs["generated_0x089000BC.class"].writeBytes(func.javaBody) }
                    println(func.disasm)
                    println("-")
                    println(func.jsbody)
                }
            }

            n += func.instructions
            func.func(cpu)
            //cpu.dump()
        } while (n < count)
        return n
    }
}

data class CpuStateFunctionCtx(val mem: Memory, val ff: CpuStateFunction, val pc: Int, val instructions: Int) {
    val func = ff.generateDynarek()
    val jsbody by lazy { ff.generateJsBody(strict = false) }
    val javaBody by lazy { ff.generateDynarekResult(1).data }
    val disasm by lazy {
        val out = arrayListOf<String>()
        for (n in 0 until instructions) {
            val rpc = pc + n * 4
            out += Disassembler.disasm(rpc, mem.lw(rpc))
        }
        out.joinToString("\n")
    }
}

class MethodCache(private val mem: Memory) {
    private val cachedFunctions = FastIntMap<CpuStateFunctionCtx>()

    fun reset() {
        cachedFunctions.clear()
    }

    fun getFunction(address: Int): CpuStateFunctionCtx {
        if (address !in cachedFunctions) {
            val dm = DynarekMethodBuilder()
            var pc = address
            var icount = 0

            do {
                dm.dispatch(pc, mem.lw(pc))
                pc += 4
                icount++
                if (icount >= 10000) error("Function too big!")
            } while (!dm.reachedFlow)
            val func = dm.generateFunction()
            cachedFunctions[address] = CpuStateFunctionCtx(mem, func, address, icount)
        }
        return cachedFunctions[address]!!
    }

    fun invalidateInstructionCache(ptr: Int, size: Int) {
        if (ptr == 0 && size == Int.MAX_VALUE) {
            cachedFunctions.clear()
        } else {
            cachedFunctions.removeRange(ptr, ptr + size - 1)
        }
    }
}