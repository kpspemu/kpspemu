package com.soywiz.kpspemu.ge

import com.soywiz.kpspemu.util.EventFlag

abstract class GpuRenderer {
	open val queuedJobs = EventFlag(0)
	abstract fun render(batches: List<GeBatchData>): Unit
	open fun reset() {
		queuedJobs.value = 0
	}
}

class DummyGpuRenderer : GpuRenderer() {
	override fun render(batches: List<GeBatchData>) {
		println("BATCHES: $batches")
	}
}