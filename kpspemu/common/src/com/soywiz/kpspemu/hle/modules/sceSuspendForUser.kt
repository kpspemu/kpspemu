package com.soywiz.kpspemu.hle.modules

import com.soywiz.kpspemu.*
import com.soywiz.kpspemu.cpu.*
import com.soywiz.kpspemu.hle.*

@Suppress("UNUSED_PARAMETER")
class sceSuspendForUser(emulator: Emulator) :
    SceModule(emulator, "sceSuspendForUser", 0x40000011, "sysmem.prx", "sceSystemMemoryManager") {
    fun sceKernelPowerLock(lockType: Int): Int {
        return 0
    }

    fun sceKernelPowerUnlock(lockType: Int): Int {
        return 0
    }

    fun sceKernelPowerTick(value: Int): Int {
        return 0
    }

    fun sceKernelVolatileMemLock(cpu: CpuState): Unit = UNIMPLEMENTED(0x3E0271D3)
    fun sceKernelVolatileMemTryLock(cpu: CpuState): Unit = UNIMPLEMENTED(0xA14F40B2)
    fun sceKernelVolatileMemUnlock(cpu: CpuState): Unit = UNIMPLEMENTED(0xA569E425)

    override fun registerModule() {
        registerFunctionInt("sceKernelPowerLock", 0xEADB1BD7, since = 150) { sceKernelPowerLock(int) }
        registerFunctionInt("sceKernelPowerUnlock", 0x3AEE7261, since = 150) { sceKernelPowerUnlock(int) }
        registerFunctionInt("sceKernelPowerTick", 0x090CCB3F, since = 150) { sceKernelPowerTick(int) }

        registerFunctionRaw("sceKernelVolatileMemLock", 0x3E0271D3, since = 150) { sceKernelVolatileMemLock(it) }
        registerFunctionRaw("sceKernelVolatileMemTryLock", 0xA14F40B2, since = 150) { sceKernelVolatileMemTryLock(it) }
        registerFunctionRaw("sceKernelVolatileMemUnlock", 0xA569E425, since = 150) { sceKernelVolatileMemUnlock(it) }
    }
}
