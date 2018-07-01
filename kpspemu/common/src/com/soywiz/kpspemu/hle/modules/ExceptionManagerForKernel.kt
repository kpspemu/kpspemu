package com.soywiz.kpspemu.hle.modules

import com.soywiz.kpspemu.*
import com.soywiz.kpspemu.cpu.*
import com.soywiz.kpspemu.hle.*

@Suppress("UNUSED_PARAMETER")
class ExceptionManagerForKernel(emulator: Emulator) :
    SceModule(emulator, "ExceptionManagerForKernel", 0x00010011, "exceptionman.prx", "sceExceptionManager") {
    fun sceKernelRegisterDefaultExceptionHandler(exceptionHandlerFunction: Int): Int {
        return 0
    }

    fun sceKernelRegisterNmiHandler(cpu: CpuState): Unit = UNIMPLEMENTED(0x15ADC862)
    fun ExceptionManagerForKernel_60DFC510(cpu: CpuState): Unit = UNIMPLEMENTED(0x60DFC510)
    fun ExceptionManagerForKernel_792C424C(cpu: CpuState): Unit = UNIMPLEMENTED(0x792C424C)
    fun ExceptionManagerForKernel_A966D178(cpu: CpuState): Unit = UNIMPLEMENTED(0xA966D178)
    fun sceKernelReleaseNmiHandler(cpu: CpuState): Unit = UNIMPLEMENTED(0xB15357C9)
    fun ExceptionManagerForKernel_CF57A486(cpu: CpuState): Unit = UNIMPLEMENTED(0xCF57A486)
    fun ExceptionManagerForKernel_D74DECBB(cpu: CpuState): Unit = UNIMPLEMENTED(0xD74DECBB)
    fun ExceptionManagerForKernel_E1F6B00B(cpu: CpuState): Unit = UNIMPLEMENTED(0xE1F6B00B)
    fun ExceptionManagerForKernel_F937D843(cpu: CpuState): Unit = UNIMPLEMENTED(0xF937D843)


    override fun registerModule() {
        registerFunctionInt(
            "sceKernelRegisterDefaultExceptionHandler",
            0x565C0B0E,
            since = 150
        ) { sceKernelRegisterDefaultExceptionHandler(int) }

        registerFunctionRaw("sceKernelRegisterNmiHandler", 0x15ADC862, since = 150) { sceKernelRegisterNmiHandler(it) }
        registerFunctionRaw(
            "ExceptionManagerForKernel_60DFC510",
            0x60DFC510,
            since = 150
        ) { ExceptionManagerForKernel_60DFC510(it) }
        registerFunctionRaw(
            "ExceptionManagerForKernel_792C424C",
            0x792C424C,
            since = 150
        ) { ExceptionManagerForKernel_792C424C(it) }
        registerFunctionRaw(
            "ExceptionManagerForKernel_A966D178",
            0xA966D178,
            since = 150
        ) { ExceptionManagerForKernel_A966D178(it) }
        registerFunctionRaw("sceKernelReleaseNmiHandler", 0xB15357C9, since = 150) { sceKernelReleaseNmiHandler(it) }
        registerFunctionRaw(
            "ExceptionManagerForKernel_CF57A486",
            0xCF57A486,
            since = 150
        ) { ExceptionManagerForKernel_CF57A486(it) }
        registerFunctionRaw(
            "ExceptionManagerForKernel_D74DECBB",
            0xD74DECBB,
            since = 150
        ) { ExceptionManagerForKernel_D74DECBB(it) }
        registerFunctionRaw(
            "ExceptionManagerForKernel_E1F6B00B",
            0xE1F6B00B,
            since = 150
        ) { ExceptionManagerForKernel_E1F6B00B(it) }
        registerFunctionRaw(
            "ExceptionManagerForKernel_F937D843",
            0xF937D843,
            since = 150
        ) { ExceptionManagerForKernel_F937D843(it) }
    }
}
