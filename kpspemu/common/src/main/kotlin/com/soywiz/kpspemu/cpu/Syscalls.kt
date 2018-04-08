package com.soywiz.kpspemu.cpu

import com.soywiz.korio.lang.*

interface Syscalls {
    fun syscall(state: CpuState, id: Int): Unit
}

class TraceSyscallHandler : Syscalls {
    override fun syscall(state: CpuState, id: Int) {
        println("%08X: Called syscall: ### %04X".format(state.PC, id))
    }
}
