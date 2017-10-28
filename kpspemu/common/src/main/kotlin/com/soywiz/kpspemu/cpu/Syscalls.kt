package com.soywiz.kpspemu.cpu

import com.soywiz.korio.ds.lmapOf
import com.soywiz.korio.lang.format

interface Syscalls {
	fun syscall(state: CpuState, id: Int): Unit
}

class TraceSyscallHandler : Syscalls {
	override fun syscall(state: CpuState, id: Int) {
		println("%08X: Called syscall: ### %04X".format(state.getPC(), id))
	}
}

class RegistrableSyscallHandler : Syscalls {
	val ids = lmapOf<Int, (CpuState, Int) -> Unit>()

	fun register(id: Int, handler: (CpuState, Int) -> Unit) {
		ids[id] = handler
	}

	fun unhandled(state: CpuState, id: Int) {
		println("%08X: Called syscall: ### %04X".format(state.getPC(), id))
	}

	override fun syscall(state: CpuState, id: Int) {
		val handler = ids[id] ?: ::unhandled
		handler(state, id)
	}
}