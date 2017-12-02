package com.soywiz.kpspemu.hle.modules

import com.soywiz.klock.Klock
import com.soywiz.korma.random.MtRand
import com.soywiz.kpspemu.Emulator
import com.soywiz.kpspemu.cpu.CpuState
import com.soywiz.kpspemu.hle.SceModule
import com.soywiz.kpspemu.mem.Ptr
import com.soywiz.kpspemu.mem.isNotNull
import com.soywiz.kpspemu.rtc
import com.soywiz.kpspemu.timeManager
import com.soywiz.kpspemu.util.currentTimeMicroDouble

@Suppress("UNUSED_PARAMETER")
open class UtilsForUser(emulator: Emulator) : UtilsBase(emulator, "UtilsForUser", 0x40010011, "sysmem.prx", "sceSystemMemoryManager") {
}
