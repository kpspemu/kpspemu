package com.soywiz.kpspemu.cpu.interpreter

import com.soywiz.korio.lang.format
import com.soywiz.kpspemu.cpu.CpuState
import com.soywiz.kpspemu.cpu.InstructionDispatcher
import com.soywiz.kpspemu.cpu.dis.disasmMacro
import com.soywiz.kpspemu.cpu.dispatch

class CpuInterpreter(val cpu: CpuState, var trace: Boolean = false) {
	val dispatcher = InstructionDispatcher(InstructionInterpreter)

	fun step() {
		if (trace) println("%08X: %s".format(cpu._PC, cpu.mem.disasmMacro(cpu._PC)))
		dispatcher.dispatch(cpu)
	}

	fun steps(count: Int) {
		for (n in 0 until count) step()
	}
}
