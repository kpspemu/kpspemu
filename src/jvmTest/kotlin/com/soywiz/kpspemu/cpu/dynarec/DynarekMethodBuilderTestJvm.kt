package com.soywiz.kpspemu.cpu.dynarec

import com.soywiz.dynarek2.*
import com.soywiz.dynarek2.target.js.*
import com.soywiz.kmem.*
import com.soywiz.korio.stream.*
import com.soywiz.kpspemu.cpu.*
import com.soywiz.kpspemu.cpu.assembler.*
import com.soywiz.kpspemu.cpu.dis.*
import com.soywiz.kpspemu.mem.*
import com.soywiz.krypto.encoding.*
import org.junit.Test
import java.io.*
import kotlin.test.*

class DynarekMethodBuilderTestJvm {
    @Test
    fun name() {

        val mb = DynarekMethodBuilder()
        val bytes = Assembler.assemble(
            "li a1, 10",
            "li a2, 0",
            "sll r0, r0, 0",
            "li a0, 0x0890_0004",
            "",
            "label1: addi a2, a2, 2",
            //"beq a1, zero, label1",
            "j label1",
            "addi a1, a1, -1",

            "bitrev a0, a0",
            "lui s0, 0x890",
            //"addiu a0, s0, 4",
            ""
        )

        for (pc in 0 until bytes.size step 4) {
            println("%08X: %s".format(pc, Disassembler.disasm(pc, bytes.readS32_le(pc))))
            mb.dispatch(pc, bytes.readS32_le(pc))
            if (mb.reachedFlow) {
                break
            }
        }

        val func = mb.generateFunction()
        val ctx = D2ContextPspEmu()

        println("----")
        println(func.generateJsBody(ctx, strict = false))

        try {
            val state = CpuState("DynarekMethodBuilderTest", GlobalCpuState(Memory()))
            val ff = func.generateCpuStateFunction(ctx)
            ff(state)
            assertEquals(0x08900004.hex, state.A0.hex)
            assertEquals(9.hex, state.A1.hex)
            assertEquals(2.hex, state.A2.hex)
            assertEquals(0x00000014.hex, state.PC.hex)
        } catch (e: InvalidCodeGenerated) {
            File("generated.class").writeBytes(e.data)
            throw e
        }

    }
}
