package com.soywiz.kpspemu.format

import com.soywiz.korio.async.syncTest
import com.soywiz.korio.stream.openSync
import com.soywiz.kpspemu.Emulator
import com.soywiz.kpspemu.embedded.Samples
import com.soywiz.kpspemu.format.elf.loadElf
import org.junit.Test
import kotlin.test.assertEquals

class ElfTest {
    @Test
    fun name() = syncTest {
        val emulator = Emulator(coroutineContext)
        val elf = emulator.loadElf(Samples.MINIFIRE_ELF.openSync())
        assertEquals(0x08900008, elf.moduleInfo.PC)
        assertEquals(0x00004821, elf.moduleInfo.GP)
    }
}