package com.soywiz.kpspemu.hle.modules

import com.soywiz.kpspemu.*
import com.soywiz.kpspemu.cpu.*
import com.soywiz.kpspemu.hle.*

class sceWlanDrv(emulator: Emulator) : SceModule(emulator, "sceWlanDrv", 0x40010011, "wlan.prx", "sceWlan_Driver") {
    fun sceWlanGetEtherAddr(cpu: CpuState): Unit = UNIMPLEMENTED(0x0C622081)
    fun sceWlanDevIsPowerOn(cpu: CpuState): Unit = UNIMPLEMENTED(0x93440B11)
    fun sceWlanGetSwitchState(cpu: CpuState): Unit = UNIMPLEMENTED(0xD7763699)


    override fun registerModule() {
        registerFunctionRaw("sceWlanGetEtherAddr", 0x0C622081, since = 150) { sceWlanGetEtherAddr(it) }
        registerFunctionRaw("sceWlanDevIsPowerOn", 0x93440B11, since = 150) { sceWlanDevIsPowerOn(it) }
        registerFunctionRaw("sceWlanGetSwitchState", 0xD7763699, since = 150) { sceWlanGetSwitchState(it) }
    }
}
