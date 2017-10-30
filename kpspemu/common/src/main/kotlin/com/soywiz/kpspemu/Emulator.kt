package com.soywiz.kpspemu

import com.soywiz.kpspemu.display.PspDisplay
import com.soywiz.kpspemu.hle.manager.*
import com.soywiz.kpspemu.mem.Memory
import com.soywiz.kpspemu.mem.ptr

class Emulator(
	val syscalls: SyscallManager = SyscallManager(),
	val mem: Memory = Memory()
) {
	val display: PspDisplay = PspDisplay(this)
	val memoryManager = MemoryManager(this)
	val threadManager = ThreadManager(this)
	val moduleManager = ModuleManager(this)
	val callbackManager = CallbackManager(this)
	val timeManager = TimeManager(this)
	val startThread = threadManager.create("_start", 0, 0, 0x1000, 0, mem.ptr(0))

	fun frameStep() {
		threadManager.vblank()
		threadManager.step()
	}
}

interface WithEmulator {
	val emulator: Emulator
}

val WithEmulator.mem: Memory get() = emulator.mem
val WithEmulator.memoryManager: MemoryManager get() = emulator.memoryManager
val WithEmulator.timeManager: TimeManager get() = emulator.timeManager
val WithEmulator.threadManager: ThreadManager get() = emulator.threadManager
val WithEmulator.callbackManager: CallbackManager get() = emulator.callbackManager
