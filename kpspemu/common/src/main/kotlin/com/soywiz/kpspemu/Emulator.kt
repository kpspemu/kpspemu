package com.soywiz.kpspemu

import com.soywiz.kpspemu.cpu.CpuState
import com.soywiz.kpspemu.cpu.RegistrableSyscallHandler
import com.soywiz.kpspemu.cpu.Syscalls
import com.soywiz.kpspemu.cpu.interpreter.CpuInterpreter
import com.soywiz.kpspemu.display.PspDisplay
import com.soywiz.kpspemu.mem.Memory

class PspThread(val mem: Memory, val syscalls: Syscalls) {
	val cpu = CpuState(mem, syscalls)
}

class Emulator(
	val syscalls: RegistrableSyscallHandler = RegistrableSyscallHandler(),
	val mem: Memory = Memory(),
	val display: PspDisplay = PspDisplay(mem)
) {
	val mainThread = PspThread(mem, syscalls)
	val cpu = mainThread.cpu
	val interpreter = CpuInterpreter(cpu)
}