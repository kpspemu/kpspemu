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
import com.soywiz.kpspemu.hle.modules.sceCtrl
import com.soywiz.kpspemu.hle.modules.sceDisplay
import com.soywiz.kpspemu.mem.Memory
import com.soywiz.kpspemu.mem.TraceMemory
import com.soywiz.kpspemu.mem.trace
import org.junit.Test

class EmulatorTest {
	@Test
	fun name() = syncTest {
		val elf = Elf.read(MinifireElf.openSync())
		//val emu = Emulator(mem = Memory().trace())
		//val emu = Emulator(mem = Memory().trace(traceReads = true))
		val emu = Emulator(mem = Memory())
		//val emu = Emulator(mem = Memory().trace(traceWrites = true))
		emu.run {
			// Hardcoded as first example
			val ph = elf.programHeaders[0]
			val programBytes = elf.stream.sliceWithSize(ph.offset.toLong(), ph.fileSize.toLong()).readAll()
			mem.write(ph.virtualAddress, programBytes)
			val moduleInfo = ElfPspModuleInfo(elf.sectionHeadersByName[".rodata.sceModuleInfo"]!!.stream.clone())
			//interpreter.trace = true
			interpreter.trace = false

			UtilsForUser().registerPspModule(emu)
			sceCtrl().registerPspModule(emu)
			sceDisplay().registerPspModule(emu)

			cpu.r29 = 0x0A000000 // stack
			cpu.setPC(0x08900008) // PC

			//for (n in 0 until 196000) {
			//for (n in 0 until 8000000) {
			for (n in 0 until 10000000) {
			//for (n in 0 until 800000) {
			//for (n in 0 until 1000000) {
				interpreter.step()
				//println("r8: ${cpu.r8}")
			}
			display.address = 0x44000000
			println("VIDEOMEM: 0x%08X".format(display.address))
			for (n in 0 until 512 * 272) {
				print(mem.lw(display.address + n * 4))
			}
			val data = mem.readBytes(display.address, 4 * 512 * 272)
			val bmp = RGBA.decodeToBitmap32(512, 272, data)
			bmp.writeTo(LocalVfs("c:/temp")["video_dump.tga"], formats = ImageFormats().registerStandard())
			bmp.writeTo(LocalVfs("c:/temp")["video_dump.png"], formats = ImageFormats().registerStandard())
		}
	}
}
