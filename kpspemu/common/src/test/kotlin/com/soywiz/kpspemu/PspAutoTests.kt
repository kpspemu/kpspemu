package com.soywiz.kpspemu

import com.soywiz.korio.async.syncTest
import com.soywiz.korio.stream.SyncStream
import com.soywiz.korio.vfs.localCurrentDirVfs
import com.soywiz.kpspemu.cpu.ExitGameException
import com.soywiz.kpspemu.format.elf.loadElfAndSetRegisters
import com.soywiz.kpspemu.hle.registerNativeModules
import org.junit.Test

class PspAutoTests {
	suspend fun testFile(elf: SyncStream) {
		val emulator = Emulator()
		emulator.display.exposeDisplay = false
		emulator.registerNativeModules()
		emulator.loadElfAndSetRegisters(elf)
		try {
			while (emulator.threadManager.aliveThreadCount >= 1) {
				emulator.frameStep()
			}
		} catch (e: ExitGameException) {
		}
		println(emulator.threadManager)
	}

	suspend fun testFile(name: String) {
		testFile(
			localCurrentDirVfs["../../pspautotests/$name.prx"].readAsSyncStream()
		)
	}

	@Test
	fun name() = syncTest { testFile("cpu/cpu_alu/cpu_alu") }
}