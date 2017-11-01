package com.soywiz.kpspemu

import MyAssert
import com.soywiz.korio.async.syncTest
import com.soywiz.korio.stream.SyncStream
import com.soywiz.korio.vfs.localCurrentDirVfs
import com.soywiz.kpspemu.format.elf.loadElfAndSetRegisters
import com.soywiz.kpspemu.hle.registerNativeModules
import org.junit.Test

class PspAutoTests {
	suspend fun testFile(elf: SyncStream, expected: String) {
		val emulator = Emulator()
		emulator.display.exposeDisplay = false
		emulator.registerNativeModules()
		emulator.loadElfAndSetRegisters(elf)
		//emulator.threadManager.trace("user_main", trace = true)
		while (emulator.threadManager.aliveThreadCount >= 1) {
			emulator.frameStep()
		}
		fun String.normalize() = this.replace("\r\n", "\n").replace("\r", "\n")
		MyAssert.assertEquals(expected.normalize(), emulator.output.toString().normalize())
	}

	fun testFile(name: String) = syncTest {
		testFile(
			localCurrentDirVfs["../../pspautotests/$name.prx"].readAsSyncStream(),
			localCurrentDirVfs["../../pspautotests/$name.expected"].readString()
		)
	}

	@Test fun testCpuAlu() = testFile("cpu/cpu_alu/cpu_alu")
	//@Test fun testFpuFpu() = testFile("cpu/fpu/fpu")
	@Test fun testCpuBranch() = testFile("cpu/cpu_alu/cpu_branch")

}