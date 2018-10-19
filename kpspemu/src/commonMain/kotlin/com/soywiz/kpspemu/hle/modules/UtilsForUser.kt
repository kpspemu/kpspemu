package com.soywiz.kpspemu.hle.modules

import com.soywiz.kpspemu.*

@Suppress("UNUSED_PARAMETER")
open class UtilsForUser(emulator: Emulator) :
    UtilsBase(emulator, "UtilsForUser", 0x40010011, "sysmem.prx", "sceSystemMemoryManager") {
}
