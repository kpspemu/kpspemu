package com.soywiz.kpspemu.ge

import com.soywiz.kpspemu.Emulator
import com.soywiz.kpspemu.WithEmulator

class Gpu(override val emulator: Emulator) : WithEmulator {
	val batchQueue = arrayListOf<GeBatch>()

	fun render() {
		if (batchQueue.isNotEmpty()) {
			println("Render: $batchQueue")
			batchQueue.clear()
		}
	}
}