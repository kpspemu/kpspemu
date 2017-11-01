package com.soywiz.kpspemu

import MyAssert
import com.soywiz.korio.async.syncTest
import com.soywiz.korio.lang.Console
import com.soywiz.korio.stream.SyncStream
import com.soywiz.korio.vfs.localCurrentDirVfs
import com.soywiz.kpspemu.format.elf.loadElfAndSetRegisters
import com.soywiz.kpspemu.hle.registerNativeModules
import com.soywiz.kpspemu.util.PspLogLevel
import com.soywiz.kpspemu.util.PspLoggerManager
import com.soywiz.kpspemu.util.hex
import com.soywiz.kpspemu.util.quote
import org.junit.Test

class IntegrationTests {
	val TRACE = false

	suspend fun testFile(elf: SyncStream, expected: String, ignores: List<String>, processor: (String) -> String = { it }) {
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

		try {
			while (emulator.running) {
				emulator.frameStep()
				if (TRACE) {
					for (thread in emulator.threadManager.threads) {
						println("PC: ${thread.state.PC.hex} : ${(thread.state.PC - info.baseAddress).hex}")
					}
				}
			}
		} catch (e: Throwable) {
			Console.error("Partial output generated:")
			Console.error("'" + emulator.output.toString() + "'")
			throw e
		}

		val ignoresRegex = ignores.map {
			Regex(Regex.quote(it).replace("\\^", ".")) to it
		}

		fun String.normalize(): String {
			var out = this.replace("\r\n", "\n").replace("\r", "\n").trimEnd()
			for (rex in ignoresRegex) {
				out = out.replace(rex.first, rex.second)
			}
			return out
		}
		MyAssert.assertEquals(expected.normalize(), processor(emulator.output.toString().normalize()))
	}

	fun testFile(name: String, ignores: List<String> = listOf(), processor: (String) -> String = { it }) = syncTest {
		testFile(
			localCurrentDirVfs["../../pspautotests/$name.prx"].readAsSyncStream(),
			localCurrentDirVfs["../../pspautotests/$name.expected"].readString(),
			ignores,
			processor
		)
	}

	@Test fun testCpuAlu() = testFile("cpu/cpu_alu/cpu_alu")
	@Test fun testCpuBranch() = testFile("cpu/cpu_alu/cpu_branch")
	@Test fun testCpuBranch2() = testFile("cpu/cpu_alu/cpu_branch2")

	@Test fun testIcache() = testFile("cpu/icache/icache")

	@Test fun testLsu() = testFile("cpu/lsu/lsu")

	@Test fun testFpu() = testFile("cpu/fpu/fpu", ignores = listOf(
		"mul.s 0.296558 * 62.000000, CAST_1 = 18.38657^",
		"mul.s 0.296558 * 62.000000, FLOOR_3 = 18.38657^"
	))

	@Test fun testFcr() = testFile("cpu/fpu/fcr", ignores = listOf(
		"Underflow:\n  fcr0: 00003351, fcr25: 00000000, fcr26: 00000000, fcr27: 00000000, fcr28: 00000000, fcr31: ^^^^^^^^",
		"Inexact:\n  fcr0: 00003351, fcr25: 00000000, fcr26: 00000000, fcr27: 00000000, fcr28: 00000000, fcr31: ^^^^^^^^"
	))

	@Test fun testRtc() = testFile("rtc/rtc")

	//@Test fun testThreadsK0() = testFile("threads/k0/k0")

	//@Test fun testVfpuColors() = testFile("cpu/vfpu/colors")

	//@Test fun testFpuFpu() = testFile("cpu/fpu/fpu")
	//@Test fun testCpuBranch() = testFile("cpu/cpu_alu/cpu_branch")
	//@Test fun testCpuBranch2() = testFile("cpu/cpu_alu/cpu_branch2")
}
