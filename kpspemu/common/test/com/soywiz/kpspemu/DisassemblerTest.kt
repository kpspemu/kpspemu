package com.soywiz.kpspemu

import com.soywiz.kpspemu.cpu.dis.*
import kotlin.test.*

class DisassemblerTest : BaseTest() {
    val dis = Disassembler
    @Test
    fun name() {
        assertEquals("syscall 0x206D", dis.disasm(0x08000000, i = 0x00081B4C)) // 4C 1B 08 00 // syscall 0x206D
        assertEquals("sll r0, r0, 0", dis.disasm(0x08000000, i = 0x00000000)) // 00 00 00 00 // nop
        assertEquals("nop", dis.disasmMacro(0x08000000, i = 0x00000000)) // 00 00 00 00 // nop
    }
}