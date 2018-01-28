package com.soywiz.kpspemu.cpu.dynarec

import com.soywiz.dynarek.generateDynarek
import com.soywiz.dynarek.js.generateJsBody
import com.soywiz.kpspemu.cpu.CpuState
import com.soywiz.kpspemu.cpu.GlobalCpuState
import com.soywiz.kpspemu.mem.Memory
import mytest.assertEquals
import org.junit.Test

class DynarekMethodBuilderTest {
    @Test
    fun name() {
        val mb = DynarekMethodBuilder()
        mb.dispatch(0, 0) // shl r0, r0, r0
        mb.dispatch(4, 0x3C100890) // lui 0x890
        mb.dispatch(8, 0x26040004) // addiu   $a0, $s0, 4
        val func = mb.generateFunction()
        println(func.generateJsBody(strict = false))

        val state = CpuState("DynarekMethodBuilderTest", GlobalCpuState(), Memory())
        val ff = func.generateDynarek()
        ff(state)
        assertEquals(0x08900004, state.r4)

    }
}