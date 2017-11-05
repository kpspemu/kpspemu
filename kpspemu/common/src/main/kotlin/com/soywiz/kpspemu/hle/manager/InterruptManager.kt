package com.soywiz.kpspemu.hle.manager

import com.soywiz.kpspemu.Emulator

class InterruptManager(val emulator: Emulator) {
	val state = emulator.globalCpuState
	fun disableAllInterrupts(): Int {
		val res = state.interruptFlags
		state.interruptFlags = 0
		return res
	}

	fun restoreInterrupts(value: Int) {
		state.interruptFlags = state.interruptFlags or value
	}
}
