package com.soywiz.kpspemu.hle.modules

import com.soywiz.kpspemu.Emulator
import com.soywiz.kpspemu.hle.SceModule

@Suppress("UNUSED_PARAMETER")
class pspDveManager(emulator: Emulator) :
    SceModule(emulator, "pspDveManager", 0x00010011, "pspDveManager.prx", "pspDveManager_Library") {

    override fun registerModule() {
    }
}
