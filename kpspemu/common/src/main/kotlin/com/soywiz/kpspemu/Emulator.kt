package com.soywiz.kpspemu

import com.soywiz.kpspemu.cpu.CpuState
import com.soywiz.kpspemu.cpu.Syscalls
import com.soywiz.kpspemu.cpu.interpreter.CpuInterpreter
import com.soywiz.kpspemu.display.PspDisplay
import com.soywiz.kpspemu.hle.manager.MemoryManager
import com.soywiz.kpspemu.hle.manager.ModuleManager
import com.soywiz.kpspemu.hle.manager.SyscallManager
import com.soywiz.kpspemu.mem.Memory

class PspThread(val mem: Memory, val syscalls: Syscalls) {
	val cpu = CpuState(mem, syscalls)
}

class Emulator(
	val syscalls: SyscallManager = SyscallManager(),
	val mem: Memory = Memory(),
	val display: PspDisplay = PspDisplay(mem)
) {
	val memoryManager = MemoryManager()
	val moduleManager = ModuleManager(this)
	val mainThread = PspThread(mem, syscalls)
	val cpu = mainThread.cpu
	val interpreter = CpuInterpreter(cpu)

	fun frameStep() {
		display.dispatchVsync()
		interpreter.steps(1000000)
	}
}