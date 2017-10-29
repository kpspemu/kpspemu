package com.soywiz.kpspemu

import com.soywiz.korim.color.RGBA
import com.soywiz.korim.format.ImageFormats
import com.soywiz.korim.format.registerStandard
import com.soywiz.korim.format.writeTo
import com.soywiz.korio.async.syncTest
import com.soywiz.korio.lang.format
import com.soywiz.korio.stream.openSync
import com.soywiz.korio.stream.readAll
import com.soywiz.korio.stream.sliceWithSize
import com.soywiz.korio.vfs.LocalVfs
import com.soywiz.kpspemu.embedded.MinifireElf
import com.soywiz.kpspemu.format.elf.Elf
import com.soywiz.kpspemu.format.elf.ElfPspModuleInfo
import com.soywiz.kpspemu.hle.modules.UtilsForUser
import com.soywiz.kpspemu.hle.modules.sceCtrl
import com.soywiz.kpspemu.hle.modules.sceDisplay
import com.soywiz.kpspemu.mem.Memory
import org.junit.Test
import kotlin.test.Ignore

class EmulatorTest {
	@Test
	@Ignore
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

			cpu.GPR[29] = 0x0A000000 // stack
			cpu.setPC(0x08900008) // PC

			var count1 = 0
			var count2 = 0

			//for (n in 0 until 196000) {
			//for (n in 0 until 80000000) {
			//for (n in 0 until 20000000) {
			for (n in 0 until 100000) {
				//for (n in 0 until 8000000) {
				//for (n in 0 until 10000000) {
				//for (n in 0 until 40000000) {
				//for (n in 0 until 800000) {
				//for (n in 0 until 1000000) {
				//val PC = cpu._PC
				interpreter.step()

				//if (PC == 0x089000AC) {
				////if (PC == 0x0890009C) {
				//	println("r2=${"%08X".format(cpu.GPR[2])}, r20=${"%08X".format(cpu.GPR[20])}")
				//}
//
				//if (PC == 0x089000B0) {
				//	println("--------------")
				//}
//
				//if (PC == 0x089000C4) {
				//	println("LO=${cpu.LO}, HI=${cpu.HI}, r2=${cpu.GPR[2]}, r3=${cpu.GPR[3]}")
				//}
//
				//if (PC == 0x089000D8) {
				//	println("r3 = ${cpu.GPR[3]}")
				//}
//
				//if (PC == 0x08900168) {
				//	println("TEST: r2(byte) = ${cpu.GPR.hex(2)}, r4(store) = ${cpu.GPR.hex(4)}, r10(read) = ${cpu.GPR.hex(10)}")
				//}

				//if (PC == 0x0890016C) {
				//	count1++
				//}
//
				//if (PC == 0x0890018C) {
				//	count2++
				//	println("r2=${cpu.r2}, r8=${cpu.r8}")
				//}
//
				//if (PC == 0x089001A0) {
				//	println("-------------- $count1, $count2")
				//	count1 = 0
				//	count2 = 0
				//}

				//println("r8: ${cpu.r8}")
			}
			/*
			display.address = 0x44000000
			println("VIDEOMEM: 0x%08X".format(display.address))
			for (n in 0 until 512 * 272) {
				print(mem.lw(display.address + n * 4))
			}
			val data = mem.readBytes(display.address, 4 * 512 * 272)
			val bmp = RGBA.decodeToBitmap32(512, 272, data)
			bmp.transformColor { RGBA(RGBA.getR(it), RGBA.getG(it), RGBA.getB(it), 0xFF) }
			bmp.writeTo(LocalVfs("c:/temp")["video_dump.tga"], formats = ImageFormats().registerStandard())
			bmp.writeTo(LocalVfs("c:/temp")["video_dump.png"], formats = ImageFormats().registerStandard())
			*/
		}
	}
}
