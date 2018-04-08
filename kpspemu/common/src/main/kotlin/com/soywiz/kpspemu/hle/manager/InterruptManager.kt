package com.soywiz.kpspemu.hle.manager

import com.soywiz.klogger.*
import com.soywiz.kpspemu.*
import com.soywiz.kpspemu.cpu.*

class InterruptManager(val emulator: Emulator) {
    val logger = Logger("InterruptManager")

    val state get() = emulator.globalCpuState

    fun disableAllInterrupts(): Int {
        val res = state.interruptFlags
        state.interruptFlags = 0
        return res
    }

    fun restoreInterrupts(value: Int) {
        state.interruptFlags = state.interruptFlags or value
    }

    data class InterruptHandler(
        val id: Int,
        var enabled: Boolean = false,
        var address: Int = 0,
        var argument: Int = 0,
        var cpuState: CpuState? = null
    ) {
        fun reset(): Unit {
            enabled = false
            address = 0
            argument = 0
            cpuState = null
        }
    }

    class InterruptKind(val index: Int) {
        val handlers by lazy { (0 until 32).map { InterruptHandler(it) } }

        fun reset() = run { for (h in handlers) h.reset() }
    }

    val interrupts by lazy { (0 until 68).map { InterruptKind(it) } }

    fun get(interrupt: Int, handlerIndex: Int): InterruptHandler {
        return interrupts[interrupt].handlers[handlerIndex]
    }

    fun dispatchVsync() = dispatchInterrupt(PspInterrupts.PSP_VBLANK_INT)
    fun isEnabled(id: Int) = (state.interruptFlags and (1 shl id)) != 0

    fun dispatchInterrupt(id: Int) {
        if (isEnabled(id)) {
            val int = interrupts[id]
            for (handler in int.handlers.filter { it.enabled }) {
                logger.trace { "Interrupt $id: $handler" }
                //logger.warn { "Interrupt $id: $handler" }
                emulator.threadManager.executeInterrupt(handler.address, handler.argument)
            }
        }
    }

    fun reset() = run { for (i in interrupts) i.reset() }
}


object PspInterrupts {
    const val PSP_GPIO_INT = 4
    const val PSP_ATA_INT = 5
    const val PSP_UMD_INT = 6
    const val PSP_MSCM0_INT = 7
    const val PSP_WLAN_INT = 8
    const val PSP_AUDIO_INT = 10
    const val PSP_I2C_INT = 12
    const val PSP_SIRCS_INT = 14
    const val PSP_SYSTIMER0_INT = 15
    const val PSP_SYSTIMER1_INT = 16
    const val PSP_SYSTIMER2_INT = 17
    const val PSP_SYSTIMER3_INT = 18
    const val PSP_THREAD0_INT = 19
    const val PSP_NAND_INT = 20
    const val PSP_DMACPLUS_INT = 21
    const val PSP_DMA0_INT = 22
    const val PSP_DMA1_INT = 23
    const val PSP_MEMLMD_INT = 24
    const val PSP_GE_INT = 25
    const val PSP_VBLANK_INT = 30 // 0x1E
    const val PSP_MECODEC_INT = 31
    const val PSP_HPREMOTE_INT = 36
    const val PSP_MSCM1_INT = 60
    const val PSP_MSCM2_INT = 61
    const val PSP_THREAD1_INT = 65
    const val PSP_INTERRUPT_INT = 66
    const val PSP_NUMBER_INTERRUPTS = 67
}

