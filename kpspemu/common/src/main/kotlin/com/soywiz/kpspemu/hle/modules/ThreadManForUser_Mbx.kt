package com.soywiz.kpspemu.hle.modules

import com.soywiz.kpspemu.cpu.*
import com.soywiz.kpspemu.hle.*

class ThreadManForUser_Mbx(val tmodule: ThreadManForUser) : SceSubmodule<ThreadManForUser>(tmodule) {
    fun sceKernelPollMbx(cpu: CpuState): Unit = UNIMPLEMENTED(0x0D81716A)
    fun sceKernelReceiveMbx(cpu: CpuState): Unit = UNIMPLEMENTED(0x18260574)
    fun sceKernelCreateMbx(cpu: CpuState): Unit = UNIMPLEMENTED(0x8125221D)
    fun sceKernelDeleteMbx(cpu: CpuState): Unit = UNIMPLEMENTED(0x86255ADA)
    fun sceKernelCancelReceiveMbx(cpu: CpuState): Unit = UNIMPLEMENTED(0x87D4DD36)
    fun sceKernelReferMbxStatus(cpu: CpuState): Unit = UNIMPLEMENTED(0xA8E8C846)
    fun sceKernelSendMbx(cpu: CpuState): Unit = UNIMPLEMENTED(0xE9B3061E)
    fun sceKernelReceiveMbxCB(cpu: CpuState): Unit = UNIMPLEMENTED(0xF3986382)

    fun registerSubmodule() = tmodule.apply {
        registerFunctionRaw("sceKernelPollMbx", 0x0D81716A, since = 150) { sceKernelPollMbx(it) }
        registerFunctionRaw("sceKernelReceiveMbx", 0x18260574, since = 150) { sceKernelReceiveMbx(it) }
        registerFunctionRaw("sceKernelCreateMbx", 0x8125221D, since = 150) { sceKernelCreateMbx(it) }
        registerFunctionRaw("sceKernelDeleteMbx", 0x86255ADA, since = 150) { sceKernelDeleteMbx(it) }
        registerFunctionRaw("sceKernelCancelReceiveMbx", 0x87D4DD36, since = 150) { sceKernelCancelReceiveMbx(it) }
        registerFunctionRaw("sceKernelReferMbxStatus", 0xA8E8C846, since = 150) { sceKernelReferMbxStatus(it) }
        registerFunctionRaw("sceKernelSendMbx", 0xE9B3061E, since = 150) { sceKernelSendMbx(it) }
        registerFunctionRaw("sceKernelReceiveMbxCB", 0xF3986382, since = 150) { sceKernelReceiveMbxCB(it) }
    }
}
