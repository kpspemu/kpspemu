package com.soywiz.kpspemu

import com.soywiz.korio.stream.openSync
import com.soywiz.korio.stream.readAll
import com.soywiz.korio.stream.sliceWithSize
import com.soywiz.kpspemu.cpu.dis.Disassembler
import com.soywiz.kpspemu.embedded.MinifireElf
import com.soywiz.kpspemu.format.Elf
import com.soywiz.kpspemu.format.ElfPspModuleInfo
import com.soywiz.kpspemu.hle.modules.UtilsForUser
import org.junit.Test

class EmulatorTest {
	@Test
	fun name() {
		val elf = Elf.read(MinifireElf.openSync())
		val emu = Emulator()

		// Hardcoded as first example
		val disasm = Disassembler
		val ph = elf.programHeaders[0]
		val programBytes = elf.stream.sliceWithSize(ph.offset.toLong(), ph.fileSize.toLong()).readAll()
		emu.mem.write(ph.virtualAddress, programBytes)
		val moduleInfo = ElfPspModuleInfo(elf.sectionHeadersByName[".rodata.sceModuleInfo"]!!.stream.clone())
		val interpreter = emu.interpreter
		interpreter.trace = true

		UtilsForUser().registerPspModule(emu)

		emu.cpu.setPC(0x08900008)
		for (n in 0 until 100) {
			interpreter.step()
		}
	}
}
