package com.soywiz.kpspemu.hle.modules

import com.soywiz.kpspemu.cpu.CpuState

interface ThreadManForUser_Fpl {
	fun ThreadManForUser.sceKernelCreateFpl(cpu: CpuState): Unit = UNIMPLEMENTED(0xC07BB470)
	fun ThreadManForUser.sceKernelFreeFpl(cpu: CpuState): Unit = UNIMPLEMENTED(0xF6414A71)
	fun ThreadManForUser.sceKernelDeleteFpl(cpu: CpuState): Unit = UNIMPLEMENTED(0xED1410E0)
	fun ThreadManForUser.sceKernelTryAllocateFpl(cpu: CpuState): Unit = UNIMPLEMENTED(0x623AE665)
	fun ThreadManForUser.sceKernelCancelFpl(cpu: CpuState): Unit = UNIMPLEMENTED(0xA8AA591F)
	fun ThreadManForUser.sceKernelReferFplStatus(cpu: CpuState): Unit = UNIMPLEMENTED(0xD8199E4C)
	fun ThreadManForUser.sceKernelAllocateFpl(cpu: CpuState): Unit = UNIMPLEMENTED(0xD979E9BF)
	fun ThreadManForUser.sceKernelAllocateFplCB(cpu: CpuState): Unit = UNIMPLEMENTED(0xE7282CB6)

	fun ThreadManForUser.registerModuleFpl() {
		registerFunctionRaw("sceKernelCreateFpl", 0xC07BB470, since = 150) { sceKernelCreateFpl(it) }

		registerFunctionRaw("sceKernelTryAllocateFpl", 0x623AE665, since = 150) { sceKernelTryAllocateFpl(it) }
		registerFunctionRaw("sceKernelCancelFpl", 0xA8AA591F, since = 150) { sceKernelCancelFpl(it) }
		registerFunctionRaw("sceKernelReferFplStatus", 0xD8199E4C, since = 150) { sceKernelReferFplStatus(it) }
		registerFunctionRaw("sceKernelAllocateFpl", 0xD979E9BF, since = 150) { sceKernelAllocateFpl(it) }
		registerFunctionRaw("sceKernelAllocateFplCB", 0xE7282CB6, since = 150) { sceKernelAllocateFplCB(it) }
		registerFunctionRaw("sceKernelDeleteFpl", 0xED1410E0, since = 150) { sceKernelDeleteFpl(it) }
		registerFunctionRaw("sceKernelFreeFpl", 0xF6414A71, since = 150) { sceKernelFreeFpl(it) }
	}
}