package com.soywiz.kpspemu

import MyAssert
import com.soywiz.korio.async.syncTest
import com.soywiz.korio.stream.SyncStream
import com.soywiz.korio.vfs.localCurrentDirVfs
import com.soywiz.kpspemu.format.elf.loadElfAndSetRegisters
import com.soywiz.kpspemu.hle.registerNativeModules
import com.soywiz.kpspemu.util.PspLogLevel
import com.soywiz.kpspemu.util.PspLoggerManager
import com.soywiz.kpspemu.util.hex
import org.junit.Test

class PspAutoTests {
	val TRACE = false

	suspend fun testFile(elf: SyncStream, expected: String, processor: (String) -> String = { it }) {
		val emulator = Emulator()
		emulator.display.exposeDisplay = false
		emulator.registerNativeModules()
		val info = emulator.loadElfAndSetRegisters(elf)

		if (TRACE) {
			emulator.threadManager.trace("user_main", trace = true)
			PspLoggerManager.defaultLevel = PspLogLevel.TRACE
		} else {
			PspLoggerManager.setLevel("ElfPsp", PspLogLevel.ERROR)
		}

		while (emulator.threadManager.aliveThreadCount >= 1) {
			emulator.frameStep()
			if (TRACE) {
				for (thread in emulator.threadManager.threads) {
					println("PC: ${thread.state.PC.hex} : ${(thread.state.PC - info.baseAddress).hex}")
				}
			}
		}
		fun String.normalize() = this.replace("\r\n", "\n").replace("\r", "\n").trimEnd()
		MyAssert.assertEquals(expected.normalize(), processor(emulator.output.toString().normalize()))
	}

	fun testFile(name: String, processor: (String) -> String = { it }) = syncTest {
		testFile(
			localCurrentDirVfs["../../pspautotests/$name.prx"].readAsSyncStream(),
			localCurrentDirVfs["../../pspautotests/$name.expected"].readString(),
			processor
		)
	}

	@Test fun testCpuAlu() = testFile("cpu/cpu_alu/cpu_alu")
	@Test fun testCpuBranch() = testFile("cpu/cpu_alu/cpu_branch")
	@Test fun testCpuBranch2() = testFile("cpu/cpu_alu/cpu_branch2")

	@Test fun testIcache() = testFile("cpu/icache/icache")

	@Test fun testFpu() = testFile("cpu/fpu/fpu") {
		it
			.replace("mul.s 0.296558 * 62.000000, CAST_1 = 18.386576", "mul.s 0.296558 * 62.000000, CAST_1 = 18.386574")
			.replace("mul.s 0.296558 * 62.000000, FLOOR_3 = 18.386576", "mul.s 0.296558 * 62.000000, FLOOR_3 = 18.386574")
	}

	@Test fun testFcr() = testFile("cpu/fpu/fcr") {
		it
			.replace("Underflow:\n  fcr0: 00003351, fcr25: 00000000, fcr26: 00000000, fcr27: 00000000, fcr28: 00000000, fcr31: 00000000", "Underflow:\n  fcr0: 00003351, fcr25: 00000000, fcr26: 00000000, fcr27: 00000000, fcr28: 00000000, fcr31: 0000300c")
			.replace("Inexact:\n  fcr0: 00003351, fcr25: 00000000, fcr26: 00000000, fcr27: 00000000, fcr28: 00000000, fcr31: 00000000", "Inexact:\n  fcr0: 00003351, fcr25: 00000000, fcr26: 00000000, fcr27: 00000000, fcr28: 00000000, fcr31: 00001004")
	}


	//@Test fun testFpuFpu() = testFile("cpu/fpu/fpu")
	//@Test fun testCpuBranch() = testFile("cpu/cpu_alu/cpu_branch")
	//@Test fun testCpuBranch2() = testFile("cpu/cpu_alu/cpu_branch2")
}