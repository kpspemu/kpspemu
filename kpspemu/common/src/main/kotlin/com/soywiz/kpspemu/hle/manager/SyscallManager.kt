package com.soywiz.kpspemu.hle.manager

import com.soywiz.korio.lang.format
import com.soywiz.kpspemu.cpu.CpuState
import com.soywiz.kpspemu.cpu.Syscalls
import com.soywiz.kpspemu.hle.NativeFunction
import com.soywiz.kpspemu.util.IntMap
import com.soywiz.kpspemu.util.PspLogger

class SyscallManager : Syscalls {
	val logger = PspLogger("SyscallManager")
	var lasSyscallId = 1

	fun unhandled(state: CpuState, id: Int) {
		println("%08X: Called syscall: ### %04X".format(state.PC, id))
	}

	val syscallToFunc = IntMap<(CpuState, Int) -> Unit>()
	val syscallToName = IntMap<String>()

	fun register(id: Int = -1, name: String, callback: (CpuState, Int) -> Unit): Int {
		val syscallId = if (id < 0) lasSyscallId++ else id
		syscallToFunc[syscallId] = callback
		syscallToName[syscallId] = name
		return syscallId
	}

	fun register(nfunc: NativeFunction, id: Int = -1): Int {
		return register(id, nfunc.name) { cpu, _ -> nfunc.function(cpu) }
	}

	override fun syscall(state: CpuState, id: Int) {
		val func = syscallToFunc[id] ?: ::unhandled
		logger.trace { "syscall: $id (${syscallToName[id]})" }
		func(state, id)
	}
}