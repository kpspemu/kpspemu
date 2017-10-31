package com.soywiz.kpspemu.ge

import com.soywiz.kpspemu.mem

class GeBatchBuilder(val ge: Ge) {
	val state = ge.state
	val mem = ge.mem
	var primBatchPrimitiveType: Int = -1
	var primitiveType: Int = -1
	var vertexType: Int = -1
	var vertexCount: Int = 0
	var indexAddress: Int = 0
	var vertexAddress: Int = 0
	var vertexSize: Int = 0

	val vertexBuffer = ByteArray(0x10000)
	var vertexBufferPos = 0
	val indexBuffer = ShortArray(0x10000)
	var indexBufferPos = 0

	fun reset() {
		primBatchPrimitiveType = -1
		primitiveType = -1
		vertexType = -1
		vertexCount = 0
		vertexBufferPos = 0
		indexBufferPos = 0
		vertexSize = 0
	}

	fun setVertexKind(primitiveType: Int, vertexType: Int) {
		if (this.primitiveType != primitiveType || this.vertexType != vertexType) flush()
		this.primitiveType = primitiveType
		this.vertexType = vertexType
		this.vertexSize = VertexType.size(vertexType)
	}

	fun tflush() {
	}

	fun tsync() {
	}

	fun flush() {
		if (vertexCount > 0) {
			ge.emitBatch(GeBatch(ge.state.clone(), primitiveType, vertexCount, vertexType, vertexBuffer.copyOf(vertexBufferPos), indexBuffer.copyOf(indexBufferPos)))
			vertexCount = 0
			vertexBufferPos = 0
			indexBufferPos = 0
		}
	}

	fun putVertex(address: Int) {
		//println("putVertex: ${address.hexx}, $vertexSize")
		// @TODO: Improve performance doing a fast copy
		for (n in 0 until vertexSize) {
			vertexBuffer[vertexBufferPos++] = mem.lb(address + n).toByte()
		}
	}

	fun addIndices(size: Int, count: Int) {
		//println("addIndices: size=$size, count=$count")
		when (size) {
			0 -> {
				var vaddr = state.vertexAddress
				for (n in 0 until count) {
					putVertex(vaddr)
					indexBuffer[indexBufferPos++] = vertexCount++.toShort()
					vaddr += vertexSize
				}
				state.vertexAddress = vaddr
			}
			else -> TODO("addIndices: $size")
		}
	}
}
