package com.soywiz.kpspemu

import com.soywiz.korim.color.RGBA
import com.soywiz.korim.format.*
import com.soywiz.korio.async.syncTest
import com.soywiz.korio.lang.format
import com.soywiz.korio.stream.openSync
import com.soywiz.korio.stream.readAll
import com.soywiz.korio.stream.sliceWithSize
import com.soywiz.korio.vfs.LocalVfs
import com.soywiz.kpspemu.embedded.MinifireElf
import com.soywiz.kpspemu.format.Elf
import com.soywiz.kpspemu.format.ElfPspModuleInfo
import com.soywiz.kpspemu.hle.modules.UtilsForUser
import com.soywiz.kpspemu.mem.Memory
import com.soywiz.kpspemu.mem.TraceMemory
import org.junit.Test

class EmulatorTest {
	@Test
	fun name() = syncTest {
		val elf = Elf.read(MinifireElf.openSync())
		val emu = Emulator(mem = TraceMemory(Memory()))
		//val emu = Emulator(mem = Memory())
		emu.run {
			// Hardcoded as first example
			val ph = elf.programHeaders[0]
			val programBytes = elf.stream.sliceWithSize(ph.offset.toLong(), ph.fileSize.toLong()).readAll()
			mem.write(ph.virtualAddress, programBytes)
			val moduleInfo = ElfPspModuleInfo(elf.sectionHeadersByName[".rodata.sceModuleInfo"]!!.stream.clone())
			//interpreter.trace = true
			interpreter.trace = false

			UtilsForUser().registerPspModule(emu)

			cpu.setPC(0x08900008)
			for (n in 0 until 8000000) {
				interpreter.step()
				//println("r8: ${cpu.r8}")
			}
			val data = mem.readBytes(Memory.VIDEOMEM.start, 4 * 480 * 272)
			println("VIDEOMEM: 0x%08X".format(Memory.VIDEOMEM.start))
			val bmp = RGBA.decodeToBitmap32(480, 272, data)
			bmp.writeTo(LocalVfs("c:/temp")["video_dump.tga"], formats = ImageFormats().registerStandard())
		}
	}
}
