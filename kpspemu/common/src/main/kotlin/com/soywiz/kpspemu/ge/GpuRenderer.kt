package com.soywiz.kpspemu.ge

interface GpuRenderer {
	fun render(batches: List<GeBatch>): Unit
}

class DummyGpuRenderer : GpuRenderer {
	override fun render(batches: List<GeBatch>) {
		println("BATCHES: $batches")
	}

}