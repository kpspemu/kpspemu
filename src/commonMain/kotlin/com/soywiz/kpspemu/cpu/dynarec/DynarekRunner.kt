package com.soywiz.kpspemu.cpu.dynarec

import com.soywiz.dynarek2.*
import com.soywiz.dynarek2.target.js.*
import com.soywiz.kds.*
import com.soywiz.kpspemu.cpu.*
import com.soywiz.kpspemu.cpu.dis.*
import com.soywiz.kpspemu.cpu.interpreter.*
import com.soywiz.kpspemu.mem.*
import com.soywiz.krypto.encoding.*

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
        val cpu = this.cpu
        val regs = cpu.registers
        val mem = cpu.mem
        val regsMem = regs.data.mem
        val memMem = mem

        var cachedFunc1: CpuStateFunctionCtx? = null
        var cachedPC1 = -1

        var cachedFunc2: CpuStateFunctionCtx? = null
        var cachedPC2 = -1

        var cachedFunc3: CpuStateFunctionCtx? = null
        var cachedPC3 = -1

        var cachedFunc4: CpuStateFunctionCtx? = null
        var cachedPC4 = -1

        val runner = D2Runner()
        try {
            runner.setParams(regsMem, mem, null, cpu)
            do {
                if (trace) {
                    cpu.dump()
                    println("####### ${regs.PC.hex}:")
                }
                val func = when {
                    regs.PC == cachedPC1 -> cachedFunc1!!
                    regs.PC == cachedPC2 -> cachedFunc2!!
                    regs.PC == cachedPC3 -> cachedFunc3!!
                    regs.PC == cachedPC4 -> cachedFunc4!!
                    else -> {
                        val func = mcache.getFunction(regs.PC)

                        cachedFunc4 = cachedFunc3
                        cachedPC4 = cachedPC3

                        cachedFunc3 = cachedFunc2
                        cachedPC3 = cachedPC2

                        cachedFunc2 = cachedFunc1
                        cachedPC2 = cachedPC1

                        cachedFunc1 = func
                        cachedPC1 = regs.PC
                        func!!
                    }
                }

                runner.setFunc(func.result)

                if (dump) {
                    if (!disasmPcs.contains(func.pc)) {
                        cpu.dump()
                        println("####### ${regs.PC.hex}:")
                        disasmPcs.add(func.pc)
                        //if (cpu.PC == 0x089000BC) runBlocking { localCurrentDirVfs["generated_0x089000BC.class"].writeBytes(func.javaBody) }
                        println(func.disasm)
                        println("-")
                        println(func.jsbody)
                    }
                }

                n += func.instructions
                val result = runner.execute()
                if (result != 0) {
                    throw CpuBreakExceptionCached(result)
                }
                //cpu.dump()
            } while (n < count)
        } finally {
            runner.close()
        }
        return n
    }
}

typealias CpuStateFunction = D2Func

data class CpuStateFunctionCtx(val mem: Memory, val ff: CpuStateFunction, val pc: Int, val instructions: Int) {
    val ctx = D2ContextPspEmu()
    //val result = ff.generate(ctx, "func_${pc.shex}", debug = true)
    val result = ff.generate(ctx, "func_${pc.shex}", debug = false)
    val func = result.generateCpuStateFunction()
    val jsbody by lazy { ff.generateJsBody(ctx, strict = false) }
    //val javaBody by lazy { ff.generateDynarekResult("func_${pc.shex}", 1).data }
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
    private val cachedFunctions = IntMap<CpuStateFunctionCtx>()

    fun reset() {
        cachedFunctions.clear()
    }

    fun getFunction(address: Int): CpuStateFunctionCtx = cachedFunctions[address] ?: createFunction(address)

    fun createFunction(address: Int): CpuStateFunctionCtx {
        val dm = DynarekMethodBuilder()
        var pc = address
        var icount = 0

        do {
            dm.dispatch(pc, mem.lw(pc))
            pc += 4
            icount++
            if (icount >= 10_000) error("Function too big!")
        } while (!dm.reachedFlow)
        val func = dm.generateFunction()
        val ff = CpuStateFunctionCtx(mem, func, address, icount)
        cachedFunctions[address] = ff
        return ff
    }

    fun invalidateInstructionCache(ptr: Int, size: Int) {
        if (ptr == 0 && size == Int.MAX_VALUE) {
            cachedFunctions.clear()
        } else {
            cachedFunctions.removeRange(ptr, ptr + size - 1)
        }
    }
}