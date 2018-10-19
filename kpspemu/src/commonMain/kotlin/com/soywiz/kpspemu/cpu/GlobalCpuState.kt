package com.soywiz.kpspemu.cpu

import com.soywiz.korge.util.*
import com.soywiz.kpspemu.cpu.dynarec.*
import com.soywiz.kpspemu.mem.*

class GlobalCpuState(val mem: Memory) {
    var insideInterrupt = false
    var interruptFlags = -1
    val mcache = MethodCache(mem)

    @NativeThreadLocal
    companion object {
        val dummy = GlobalCpuState(MemoryInfo.DUMMY)
    }

    fun reset() {
        mcache.reset()
        insideInterrupt = false
        interruptFlags = -1
    }
}
