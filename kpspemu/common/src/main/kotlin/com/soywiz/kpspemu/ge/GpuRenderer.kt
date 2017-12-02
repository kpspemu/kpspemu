package com.soywiz.kpspemu.ge

interface GpuRenderer {
	fun render(batches: List<GeBatchData>): Unit
	fun reset() = Unit
}

class DummyGpuRenderer : GpuRenderer {
	override fun render(batches: List<GeBatchData>) {
		println("BATCHES: $batches")
	}
}