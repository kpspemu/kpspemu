package com.soywiz.kpspemu.ge

import com.soywiz.kpspemu.util.*

abstract class GpuRenderer {
    open val queuedJobs = EventFlag(0)
    abstract fun render(batches: List<GeBatchData>): Unit
    open fun reset() = run { queuedJobs.value = 0 }
    open fun tryExecuteNow() = Unit
}

class DummyGpuRenderer : GpuRenderer() {
    override fun render(batches: List<GeBatchData>) {
        println("BATCHES: $batches")
    }
}