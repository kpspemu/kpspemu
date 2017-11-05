package com.soywiz.kpspemu.hle.modules

import com.soywiz.kpspemu.Emulator
import com.soywiz.kpspemu.cpu.CpuState
import com.soywiz.kpspemu.hle.SceModule

@Suppress("UNUSED_PARAMETER")
class InterruptManager(emulator: Emulator) : SceModule(emulator, "InterruptManager", 0x40000011, "interruptman.prx", "sceInterruptManager") {
	fun sceKernelSuspendSubIntr(cpu: CpuState): Unit = UNIMPLEMENTED(0x5CB5A78B)
	fun sceKernelResumeSubIntr(cpu: CpuState): Unit = UNIMPLEMENTED(0x7860E0DC)
	fun sceKernelDisableSubIntr(cpu: CpuState): Unit = UNIMPLEMENTED(0x8A389411)
	fun sceKernelRegisterSubIntrHandler(cpu: CpuState): Unit = UNIMPLEMENTED(0xCA04A2B9)
	fun QueryIntrHandlerInfo(cpu: CpuState): Unit = UNIMPLEMENTED(0xD2E8363F)
	fun sceKernelReleaseSubIntrHandler(cpu: CpuState): Unit = UNIMPLEMENTED(0xD61E6961)
	fun sceKernelRegisterUserSpaceIntrStack(cpu: CpuState): Unit = UNIMPLEMENTED(0xEEE43F47)
	fun sceKernelEnableSubIntr(cpu: CpuState): Unit = UNIMPLEMENTED(0xFB8E22EC)
	fun sceKernelIsSubInterruptOccurred(cpu: CpuState): Unit = UNIMPLEMENTED(0xFC4374B8)


	override fun registerModule() {
		registerFunctionRaw("sceKernelSuspendSubIntr", 0x5CB5A78B, since = 150) { sceKernelSuspendSubIntr(it) }
		registerFunctionRaw("sceKernelResumeSubIntr", 0x7860E0DC, since = 150) { sceKernelResumeSubIntr(it) }
		registerFunctionRaw("sceKernelDisableSubIntr", 0x8A389411, since = 150) { sceKernelDisableSubIntr(it) }
		registerFunctionRaw("sceKernelRegisterSubIntrHandler", 0xCA04A2B9, since = 150) { sceKernelRegisterSubIntrHandler(it) }
		registerFunctionRaw("QueryIntrHandlerInfo", 0xD2E8363F, since = 150) { QueryIntrHandlerInfo(it) }
		registerFunctionRaw("sceKernelReleaseSubIntrHandler", 0xD61E6961, since = 150) { sceKernelReleaseSubIntrHandler(it) }
		registerFunctionRaw("sceKernelRegisterUserSpaceIntrStack", 0xEEE43F47, since = 150) { sceKernelRegisterUserSpaceIntrStack(it) }
		registerFunctionRaw("sceKernelEnableSubIntr", 0xFB8E22EC, since = 150) { sceKernelEnableSubIntr(it) }
		registerFunctionRaw("sceKernelIsSubInterruptOccurred", 0xFC4374B8, since = 150) { sceKernelIsSubInterruptOccurred(it) }
	}
}
