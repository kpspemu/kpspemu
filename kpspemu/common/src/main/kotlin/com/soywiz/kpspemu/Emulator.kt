package com.soywiz.kpspemu

import com.soywiz.kpspemu.cpu.interpreter.CpuInterpreter
import com.soywiz.kpspemu.display.PspDisplay
import com.soywiz.kpspemu.hle.manager.MemoryManager
import com.soywiz.kpspemu.hle.manager.ModuleManager
import com.soywiz.kpspemu.hle.manager.SyscallManager
import com.soywiz.kpspemu.hle.manager.ThreadManager
import com.soywiz.kpspemu.mem.Memory
import com.soywiz.kpspemu.mem.ptr

class Emulator(
	val syscalls: SyscallManager = SyscallManager(),
	val mem: Memory = Memory(),
	val display: PspDisplay = PspDisplay(mem)
) {
	val memoryManager = MemoryManager()
	val threadManager = ThreadManager(this)
	val moduleManager = ModuleManager(this)
	val startThread = threadManager.createThread("_start", 0, 0, 0x1000, 0, mem.ptr(0))

	fun frameStep() {
		display.dispatchVsync()
		threadManager.step()
	}
}

interface WithEmulator {
	val emulator: Emulator
}

val WithEmulator.mem: Memory get() = emulator.mem
