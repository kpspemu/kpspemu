package com.soywiz.kpspemu.cpu

class GlobalCpuState {
	var insideInterrupt = false
	var interruptFlags = -1

	companion object {
		val dummy = GlobalCpuState()
	}

	fun reset() {
		insideInterrupt = false
		interruptFlags = -1
	}
}
