package com.soywiz.kpspemu.hle.modules


import com.soywiz.kpspemu.Emulator
import com.soywiz.kpspemu.cpu.CpuState
import com.soywiz.kpspemu.hle.SceModule


class sceDmac(emulator: Emulator) : SceModule(emulator, "sceDmac", 0x40010011, "lowio.prx", "sceLowIO_Driver") {
	fun sceDmacMemcpy(cpu: CpuState): Unit = UNIMPLEMENTED(0x617F3FE6)
	fun sceDmacTryMemcpy(cpu: CpuState): Unit = UNIMPLEMENTED(0xD97F94D8)


	override fun registerModule() {
		registerFunctionRaw("sceDmacMemcpy", 0x617F3FE6, since = 150) { sceDmacMemcpy(it) }
		registerFunctionRaw("sceDmacTryMemcpy", 0xD97F94D8, since = 150) { sceDmacTryMemcpy(it) }
	}
}
