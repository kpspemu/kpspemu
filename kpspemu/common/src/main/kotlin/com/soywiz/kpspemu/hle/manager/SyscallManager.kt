package com.soywiz.kpspemu.hle.manager

import com.soywiz.korio.lang.format
import com.soywiz.kpspemu.cpu.CpuState
import com.soywiz.kpspemu.cpu.Syscalls
import com.soywiz.kpspemu.hle.modules.NativeFunction

class SyscallManager : Syscalls {
	var lasSyscallId = 1

	fun unhandled(state: CpuState, id: Int) {
		println("%08X: Called syscall: ### %04X".format(state.getPC(), id))
	}

	val syscallToFunc = LinkedHashMap<Int, (CpuState, Int) -> Unit>()

	fun register(id: Int = -1, callback: (CpuState, Int) -> Unit): Int {
		val syscallId = if (id < 0) lasSyscallId++ else id
		syscallToFunc[syscallId] = callback
		return syscallId
	}

	fun register(nfunc: NativeFunction, id: Int = -1): Int {
		return register(id) { cpu, _ -> nfunc.function(cpu) }
	}

	override fun syscall(state: CpuState, id: Int) {
		val func = syscallToFunc.getOrElse(id) { ::unhandled }
		func(state, id)
	}
}