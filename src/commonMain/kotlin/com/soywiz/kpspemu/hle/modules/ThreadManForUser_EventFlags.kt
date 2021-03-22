package com.soywiz.kpspemu.hle.modules

import com.soywiz.kpspemu.cpu.*
import com.soywiz.kpspemu.hle.*
import com.soywiz.kpspemu.hle.error.*
import com.soywiz.kpspemu.hle.manager.*
import com.soywiz.kpspemu.mem.*
import com.soywiz.kpspemu.util.*

class ThreadManForUser_EventFlags(val tmodule: ThreadManForUser) : SceSubmodule<ThreadManForUser>(tmodule) {
    val eventFlags = ResourceList("EventFlag") { PspEventFlag(it) }

    fun sceKernelCreateEventFlag(name: String?, attributes: Int, bitPattern: Int, optionsPtr: Ptr): Int {
        return eventFlags.alloc().apply {
            this.name = name ?: "eventFlag"
            this.attributes = attributes
            this.currentPattern = bitPattern
            this.optionsPtr = optionsPtr
        }.id
    }

    private fun getEventFlag(id: Int): PspEventFlag =
        eventFlags.tryGetById(id) ?: sceKernelException(SceKernelErrors.ERROR_KERNEL_NOT_FOUND_EVENT_FLAG)

    fun sceKernelPollEventFlag(id: Int, bits: Int, waitType: Int, outBits: Ptr): Int {
        val eventFlag = getEventFlag(id)
        if ((waitType and EventFlagWaitTypeSet.MaskValidBits.inv()) != 0) return SceKernelErrors.ERROR_KERNEL_ILLEGAL_MODE
        if ((waitType and (EventFlagWaitTypeSet.Clear or EventFlagWaitTypeSet.ClearAll)) == (EventFlagWaitTypeSet.Clear or EventFlagWaitTypeSet.ClearAll)) {
            return SceKernelErrors.ERROR_KERNEL_ILLEGAL_MODE
        }
        if (bits == 0) return SceKernelErrors.ERROR_KERNEL_EVENT_FLAG_ILLEGAL_WAIT_PATTERN
        //if (EventFlag == null) return SceKernelErrors.ERROR_KERNEL_NOT_FOUND_EVENT_FLAG;

        val matched = eventFlag.poll(bits, waitType, outBits)

        return if (matched) 0 else SceKernelErrors.ERROR_KERNEL_EVENT_FLAG_POLL_FAILED
    }

    fun sceKernelSetEventFlag(id: Int, bitPattern: Int): Int {
        val eventFlag = getEventFlag(id)
        eventFlag.setBits(bitPattern)
        return 0
    }

    fun sceKernelDeleteEventFlag(id: Int): Int {
        getEventFlag(id) // To throw exceptions if not exists
        eventFlags.freeById(id)
        return 0
    }

    fun sceKernelCancelEventFlag(cpu: CpuState): Unit = UNIMPLEMENTED(0xCD203292)
    fun sceKernelReferEventFlagStatus(cpu: CpuState): Unit = UNIMPLEMENTED(0xA66B0120)
    fun sceKernelWaitEventFlagCB(cpu: CpuState): Unit = UNIMPLEMENTED(0x328C546A)
    fun sceKernelWaitEventFlag(cpu: CpuState): Unit = UNIMPLEMENTED(0x402FCF22)
    fun sceKernelClearEventFlag(cpu: CpuState): Unit = UNIMPLEMENTED(0x812346E4)

    fun registerSubmodule() = tmodule.apply {
        registerFunctionInt("sceKernelCreateEventFlag", 0x55C20A00, since = 150) {
            sceKernelCreateEventFlag(
                str,
                int,
                int,
                ptr
            )
        }
        registerFunctionInt("sceKernelPollEventFlag", 0x30FD48F0, since = 150) {
            sceKernelPollEventFlag(
                int,
                int,
                int,
                ptr
            )
        }
        registerFunctionInt("sceKernelSetEventFlag", 0x1FB15A32, since = 150) { sceKernelSetEventFlag(int, int) }
        registerFunctionInt("sceKernelDeleteEventFlag", 0xEF9E4C70, since = 150) { sceKernelDeleteEventFlag(int) }

        registerFunctionRaw("sceKernelCancelEventFlag", 0xCD203292, since = 150) { sceKernelCancelEventFlag(it) }
        registerFunctionRaw(
            "sceKernelReferEventFlagStatus",
            0xA66B0120,
            since = 150
        ) { sceKernelReferEventFlagStatus(it) }
        registerFunctionRaw("sceKernelWaitEventFlagCB", 0x328C546A, since = 150) { sceKernelWaitEventFlagCB(it) }
        registerFunctionRaw("sceKernelWaitEventFlag", 0x402FCF22, since = 150) { sceKernelWaitEventFlag(it) }
        registerFunctionRaw("sceKernelClearEventFlag", 0x812346E4, since = 150) { sceKernelClearEventFlag(it) }
    }
}