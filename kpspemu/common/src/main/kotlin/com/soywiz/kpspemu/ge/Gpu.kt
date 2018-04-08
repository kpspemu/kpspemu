package com.soywiz.kpspemu.ge

import com.soywiz.kpspemu.*

class Gpu(override val emulator: Emulator) : WithEmulator {
    val batchQueue = arrayListOf<GeBatchData>()

    fun flush() {
        if (batchQueue.isNotEmpty()) {
            emulator.gpuRenderer.render(batchQueue.toList())
            batchQueue.clear()
        }
    }

    fun addBatch(batch: GeBatchData) {
        batchQueue += batch
    }

    fun reset() {
        batchQueue.clear()
    }
}