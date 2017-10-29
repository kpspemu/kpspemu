package com.soywiz.kpspemu.hle.modules


import com.soywiz.kpspemu.Emulator
import com.soywiz.kpspemu.cpu.CpuState
import com.soywiz.kpspemu.hle.SceModule


@Suppress("UNUSED_PARAMETER")
class StdioForUser(emulator: Emulator) : SceModule(emulator, "StdioForUser", 0x40010011, "iofilemgr.prx", "sceIOFileManager") {
	fun sceKernelStdin(): Int = 1
	fun sceKernelStdout(): Int = 2
	fun sceKernelStderr(): Int = 3

	fun sceKernelStdioLseek(cpu: CpuState): Unit = UNIMPLEMENTED(0x0CBB0571)
	fun sceKernelStdioRead(cpu: CpuState): Unit = UNIMPLEMENTED(0x3054D478)
	fun sceKernelRegisterStdoutPipe(cpu: CpuState): Unit = UNIMPLEMENTED(0x432D8F5C)
	fun sceKernelRegisterStderrPipe(cpu: CpuState): Unit = UNIMPLEMENTED(0x6F797E03)
	fun sceKernelStdioOpen(cpu: CpuState): Unit = UNIMPLEMENTED(0x924ABA61)
	fun sceKernelStdioClose(cpu: CpuState): Unit = UNIMPLEMENTED(0x9D061C19)
	fun sceKernelStdioWrite(cpu: CpuState): Unit = UNIMPLEMENTED(0xA3B931DB)
	fun sceKernelStdioSendChar(cpu: CpuState): Unit = UNIMPLEMENTED(0xA46785C9)


	override fun registerModule() {
		registerFunctionInt("sceKernelStdin", 0x172D316E, since = 150) { sceKernelStdin() }
		registerFunctionInt("sceKernelStdout", 0xA6BAB2E9, since = 150) { sceKernelStdout() }
		registerFunctionInt("sceKernelStderr", 0xF78BA90A, since = 150) { sceKernelStderr() }

		registerFunctionRaw("sceKernelStdioLseek", 0x0CBB0571, since = 150) { sceKernelStdioLseek(it) }
		registerFunctionRaw("sceKernelStdioRead", 0x3054D478, since = 150) { sceKernelStdioRead(it) }
		registerFunctionRaw("sceKernelRegisterStdoutPipe", 0x432D8F5C, since = 150) { sceKernelRegisterStdoutPipe(it) }
		registerFunctionRaw("sceKernelRegisterStderrPipe", 0x6F797E03, since = 150) { sceKernelRegisterStderrPipe(it) }
		registerFunctionRaw("sceKernelStdioOpen", 0x924ABA61, since = 150) { sceKernelStdioOpen(it) }
		registerFunctionRaw("sceKernelStdioClose", 0x9D061C19, since = 150) { sceKernelStdioClose(it) }
		registerFunctionRaw("sceKernelStdioWrite", 0xA3B931DB, since = 150) { sceKernelStdioWrite(it) }
		registerFunctionRaw("sceKernelStdioSendChar", 0xA46785C9, since = 150) { sceKernelStdioSendChar(it) }
	}
}
