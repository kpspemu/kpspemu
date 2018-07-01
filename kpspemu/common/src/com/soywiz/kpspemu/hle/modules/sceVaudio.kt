package com.soywiz.kpspemu.hle.modules

import com.soywiz.kpspemu.*
import com.soywiz.kpspemu.cpu.*
import com.soywiz.kpspemu.hle.*

@Suppress("UNUSED_PARAMETER")
class sceVaudio(emulator: Emulator) : SceModule(emulator, "sceVaudio", 0x40010011, "vaudio.prx", "sceVaudio_driver") {
    fun sceVaudioChReserve(cpu: CpuState): Unit = UNIMPLEMENTED(0x03B6807D)
    fun sceVaudio_27ACC20B(cpu: CpuState): Unit = UNIMPLEMENTED(0x27ACC20B)
    fun sceVaudio_346FBE94(cpu: CpuState): Unit = UNIMPLEMENTED(0x346FBE94)
    fun sceVaudio_504E4745(cpu: CpuState): Unit = UNIMPLEMENTED(0x504E4745)
    fun sceVaudioChRelease(cpu: CpuState): Unit = UNIMPLEMENTED(0x67585DFD)
    fun sceVaudioOutputBlocking(cpu: CpuState): Unit = UNIMPLEMENTED(0x8986295E)
    fun sceVaudio_CBD4AC51(cpu: CpuState): Unit = UNIMPLEMENTED(0xCBD4AC51)
    fun sceVaudio_E8E78DC8(cpu: CpuState): Unit = UNIMPLEMENTED(0xE8E78DC8)


    override fun registerModule() {
        registerFunctionRaw("sceVaudioChReserve", 0x03B6807D, since = 150) { sceVaudioChReserve(it) }
        registerFunctionRaw("sceVaudio_27ACC20B", 0x27ACC20B, since = 150) { sceVaudio_27ACC20B(it) }
        registerFunctionRaw("sceVaudio_346FBE94", 0x346FBE94, since = 150) { sceVaudio_346FBE94(it) }
        registerFunctionRaw("sceVaudio_504E4745", 0x504E4745, since = 150) { sceVaudio_504E4745(it) }
        registerFunctionRaw("sceVaudioChRelease", 0x67585DFD, since = 150) { sceVaudioChRelease(it) }
        registerFunctionRaw("sceVaudioOutputBlocking", 0x8986295E, since = 150) { sceVaudioOutputBlocking(it) }
        registerFunctionRaw("sceVaudio_CBD4AC51", 0xCBD4AC51, since = 150) { sceVaudio_CBD4AC51(it) }
        registerFunctionRaw("sceVaudio_E8E78DC8", 0xE8E78DC8, since = 150) { sceVaudio_E8E78DC8(it) }
    }
}
