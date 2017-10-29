package com.soywiz.kpspemu.format

import com.soywiz.korio.async.syncTest
import com.soywiz.korio.lang.format
import com.soywiz.korio.stream.openSync
import com.soywiz.kpspemu.Emulator
import com.soywiz.kpspemu.embedded.Samples
import com.soywiz.kpspemu.format.elf.loadElf
import org.junit.Test

class ElfTest {
	@Test
	fun name() = syncTest {
		val emulator = Emulator()
		val elf = emulator.loadElf(Samples.MINIFIRE_ELF.openSync())
		println("%08X".format(elf.moduleInfo.PC))
	}
}