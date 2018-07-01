package com.soywiz.kpspemu.hle.modules

import com.soywiz.kpspemu.*
import com.soywiz.kpspemu.cpu.*
import com.soywiz.kpspemu.hle.*
import com.soywiz.kpspemu.hle.manager.*

@Suppress("UNUSED_PARAMETER", "MemberVisibilityCanPrivate")
class InterruptManager(emulator: Emulator) :
    SceModule(emulator, "InterruptManager", 0x40000011, "interruptman.prx", "sceInterruptManager") {
    val interruptManager = emulator.interruptManager

    fun sceKernelRegisterSubIntrHandler(
        thread: PspThread,
        interrupt: Int,
        handlerIndex: Int,
        callbackAddress: Int,
        callbackArgument: Int
    ): Int {
        val intr = interruptManager.get(interrupt, handlerIndex)
        intr.address = callbackAddress
        intr.argument = callbackArgument
        intr.enabled = true
        intr.cpuState = thread.state
        return 0
    }

    fun sceKernelEnableSubIntr(interrupt: Int, handlerIndex: Int): Int {
        val intr = interruptManager.get(interrupt, handlerIndex)
        intr.enabled = true
        return 0
    }

    fun sceKernelDisableSubIntr(interrupt: Int, handlerIndex: Int): Int {
        val intr = interruptManager.get(interrupt, handlerIndex)
        intr.enabled = false
        return 0
    }

    fun sceKernelSuspendSubIntr(cpu: CpuState): Unit = UNIMPLEMENTED(0x5CB5A78B)
    fun sceKernelResumeSubIntr(cpu: CpuState): Unit = UNIMPLEMENTED(0x7860E0DC)
    fun QueryIntrHandlerInfo(cpu: CpuState): Unit = UNIMPLEMENTED(0xD2E8363F)
    fun sceKernelReleaseSubIntrHandler(cpu: CpuState): Unit = UNIMPLEMENTED(0xD61E6961)
    fun sceKernelRegisterUserSpaceIntrStack(cpu: CpuState): Unit = UNIMPLEMENTED(0xEEE43F47)
    fun sceKernelIsSubInterruptOccurred(cpu: CpuState): Unit = UNIMPLEMENTED(0xFC4374B8)


    override fun registerModule() {
        registerFunctionInt(
            "sceKernelRegisterSubIntrHandler",
            0xCA04A2B9,
            since = 150
        ) { sceKernelRegisterSubIntrHandler(thread, int, int, int, int) }
        registerFunctionInt("sceKernelEnableSubIntr", 0xFB8E22EC, since = 150) { sceKernelEnableSubIntr(int, int) }
        registerFunctionInt("sceKernelDisableSubIntr", 0x8A389411, since = 150) { sceKernelDisableSubIntr(int, int) }

        registerFunctionRaw("sceKernelSuspendSubIntr", 0x5CB5A78B, since = 150) { sceKernelSuspendSubIntr(it) }
        registerFunctionRaw("sceKernelResumeSubIntr", 0x7860E0DC, since = 150) { sceKernelResumeSubIntr(it) }
        registerFunctionRaw("QueryIntrHandlerInfo", 0xD2E8363F, since = 150) { QueryIntrHandlerInfo(it) }
        registerFunctionRaw("sceKernelReleaseSubIntrHandler", 0xD61E6961, since = 150) {
            sceKernelReleaseSubIntrHandler(
                it
            )
        }
        registerFunctionRaw(
            "sceKernelRegisterUserSpaceIntrStack",
            0xEEE43F47,
            since = 150
        ) { sceKernelRegisterUserSpaceIntrStack(it) }
        registerFunctionRaw(
            "sceKernelIsSubInterruptOccurred",
            0xFC4374B8,
            since = 150
        ) { sceKernelIsSubInterruptOccurred(it) }
    }
}
