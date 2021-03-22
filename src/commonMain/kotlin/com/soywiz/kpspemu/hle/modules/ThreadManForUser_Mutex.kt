package com.soywiz.kpspemu.hle.modules

import com.soywiz.kpspemu.cpu.*
import com.soywiz.kpspemu.hle.*

class ThreadManForUser_Mutex(val tmodule: ThreadManForUser) : SceSubmodule<ThreadManForUser>(tmodule) {
    fun sceKernelTryLockMutex(cpu: CpuState): Unit = UNIMPLEMENTED(0x0DDCD2C9)
    fun sceKernelCreateLwMutex(cpu: CpuState): Unit = UNIMPLEMENTED(0x19CFF145)
    fun sceKernelDeleteMutex(cpu: CpuState): Unit = UNIMPLEMENTED(0xF8170FBE)
    fun sceKernelReferLwMutexStatusByID(cpu: CpuState): Unit = UNIMPLEMENTED(0x4C145944)
    fun sceKernelLockMutexCB(cpu: CpuState): Unit = UNIMPLEMENTED(0x5BF4DD27)
    fun sceKernelDeleteLwMutex(cpu: CpuState): Unit = UNIMPLEMENTED(0x60107536)
    fun sceKernelUnlockMutex(cpu: CpuState): Unit = UNIMPLEMENTED(0x6B30100F)
    fun sceKernelCancelMutex(cpu: CpuState): Unit = UNIMPLEMENTED(0x87D9223C)
    fun sceKernelReferMutexStatus(cpu: CpuState): Unit = UNIMPLEMENTED(0xA9C2CB9A)
    fun sceKernelLockMutex(cpu: CpuState): Unit = UNIMPLEMENTED(0xB011B11F)
    fun sceKernelCreateMutex(cpu: CpuState): Unit = UNIMPLEMENTED(0xB7D098C6)

    fun registerSubmodule() = tmodule.apply {
        registerFunctionRaw("sceKernelTryLockMutex", 0x0DDCD2C9, since = 150) { sceKernelTryLockMutex(it) }
        registerFunctionRaw("sceKernelCreateLwMutex", 0x19CFF145, since = 150) { sceKernelCreateLwMutex(it) }
        registerFunctionRaw(
            "sceKernelReferLwMutexStatusByID",
            0x4C145944,
            since = 150
        ) { sceKernelReferLwMutexStatusByID(it) }
        registerFunctionRaw("sceKernelLockMutexCB", 0x5BF4DD27, since = 150) { sceKernelLockMutexCB(it) }
        registerFunctionRaw("sceKernelDeleteLwMutex", 0x60107536, since = 150) { sceKernelDeleteLwMutex(it) }
        registerFunctionRaw("sceKernelUnlockMutex", 0x6B30100F, since = 150) { sceKernelUnlockMutex(it) }
        registerFunctionRaw("sceKernelCancelMutex", 0x87D9223C, since = 150) { sceKernelCancelMutex(it) }
        registerFunctionRaw("sceKernelReferMutexStatus", 0xA9C2CB9A, since = 150) { sceKernelReferMutexStatus(it) }
        registerFunctionRaw("sceKernelLockMutex", 0xB011B11F, since = 150) { sceKernelLockMutex(it) }
        registerFunctionRaw("sceKernelCreateMutex", 0xB7D098C6, since = 150) { sceKernelCreateMutex(it) }
        registerFunctionRaw("sceKernelDeleteMutex", 0xF8170FBE, since = 150) { sceKernelDeleteMutex(it) }
    }
}
