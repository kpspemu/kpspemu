package com.soywiz.kpspemu.hle.modules

import com.soywiz.kpspemu.*
import com.soywiz.kpspemu.cpu.*
import com.soywiz.kpspemu.hle.*

@Suppress("UNUSED_PARAMETER")
class Kernel_Library(emulator: Emulator) :
    SceModule(emulator, "Kernel_Library", 0x00010011, "usersystemlib.prx", "sceKernelLibrary") {
    fun sceKernelCpuSuspendIntr(): Int {
        return emulator.interruptManager.disableAllInterrupts()
    }

    fun sceKernelCpuResumeIntr(value: Int): Int {
        emulator.interruptManager.restoreInterrupts(value)
        return 0
    }

    fun sceKernelUnlockLwMutex(cpu: CpuState): Unit = UNIMPLEMENTED(0x15B6446B)
    fun sceKernelLockLwMutexCB(cpu: CpuState): Unit = UNIMPLEMENTED(0x1FC64E09)
    fun sceKernelCpuResumeIntrWithSync(cpu: CpuState): Unit = UNIMPLEMENTED(0x3B84732D)
    fun sceKernelIsCpuIntrSuspended(cpu: CpuState): Unit = UNIMPLEMENTED(0x47A0B729)
    fun sceKernelIsCpuIntrEnable(cpu: CpuState): Unit = UNIMPLEMENTED(0xB55249D2)
    fun sceKernelLockLwMutex(cpu: CpuState): Unit = UNIMPLEMENTED(0xBEA46419)
    fun sceKernelReferLwMutexStatus(cpu: CpuState): Unit = UNIMPLEMENTED(0xC1734599)
    fun sceKernelTryLockLwMutex(cpu: CpuState): Unit = UNIMPLEMENTED(0xDC692EE3)


    override fun registerModule() {
        registerFunctionInt("sceKernelCpuSuspendIntr", 0x092968F4, since = 150) { sceKernelCpuSuspendIntr() }
        registerFunctionInt("sceKernelCpuResumeIntr", 0x5F10D406, since = 150) { sceKernelCpuResumeIntr(int) }

        registerFunctionRaw("sceKernelUnlockLwMutex", 0x15B6446B, since = 150) { sceKernelUnlockLwMutex(it) }
        registerFunctionRaw("sceKernelLockLwMutexCB", 0x1FC64E09, since = 150) { sceKernelLockLwMutexCB(it) }
        registerFunctionRaw("sceKernelCpuResumeIntrWithSync", 0x3B84732D, since = 150) {
            sceKernelCpuResumeIntrWithSync(
                it
            )
        }
        registerFunctionRaw("sceKernelIsCpuIntrSuspended", 0x47A0B729, since = 150) { sceKernelIsCpuIntrSuspended(it) }
        registerFunctionRaw("sceKernelIsCpuIntrEnable", 0xB55249D2, since = 150) { sceKernelIsCpuIntrEnable(it) }
        registerFunctionRaw("sceKernelLockLwMutex", 0xBEA46419, since = 150) { sceKernelLockLwMutex(it) }
        registerFunctionRaw("sceKernelReferLwMutexStatus", 0xC1734599, since = 150) { sceKernelReferLwMutexStatus(it) }
        registerFunctionRaw("sceKernelTryLockLwMutex", 0xDC692EE3, since = 150) { sceKernelTryLockLwMutex(it) }
    }
}
