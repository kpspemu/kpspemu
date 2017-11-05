package com.soywiz.kpspemu.ge

import com.soywiz.korio.typedarray.copyRangeTo
import com.soywiz.kpspemu.mem
import kotlin.math.max

class GeBatchBuilder(val ge: Ge) {
	val state = ge.state
	val mem = ge.mem
	var primBatchPrimitiveType: Int = -1
	var primitiveType: PrimitiveType? = null
	var vertexType: VertexType = VertexType(-1)
	var vertexCount: Int = 0
	var vertexSize: Int = 0

	val vertexBuffer = ByteArray(0x10000 * 16)
	var vertexBufferPos = 0
	val indexBuffer = ShortArray(0x10000)
	var indexBufferPos = 0

	fun reset() {
		//println("reset")
		flush()
		primBatchPrimitiveType = -1
		primitiveType = null
		vertexType.v = -1
		vertexCount = 0
		vertexBufferPos = 0
		indexBufferPos = 0
		vertexSize = 0
	}

	fun setVertexKind(primitiveType: PrimitiveType, state: GeState) {
		if (this.primitiveType != primitiveType || this.vertexType.v != state.vertexType) flush()
		vertexType.init(state)
		this.primitiveType = primitiveType
		this.vertexSize = vertexType.size
	}

	var texVersion = 0

	fun tflush() {
		texVersion++
		//println("tflush")
	}

	fun tsync() {
		//println("tsync")
	}

	fun flush() {
		//println("flush: $indexBufferPos")
		if (indexBufferPos > 0) {
			ge.emitBatch(GeBatchData(ge.state.data.copyOf(), primitiveType ?: PrimitiveType.TRIANGLES, indexBufferPos, vertexBuffer.copyOf(vertexBufferPos), indexBuffer.copyOf(indexBufferPos), texVersion))
			vertexCount = 0
			vertexBufferPos = 0
			indexBufferPos = 0
		}
	}

	fun putVertex(address: Int) {
		//println("putVertex: ${address.hexx}, $vertexSize")
		mem.read(address, vertexBuffer, vertexBufferPos, vertexSize)
		vertexBufferPos += vertexSize
		vertexCount++
	}

	fun putIndex(index: Int) {
		indexBuffer[indexBufferPos++] = index.toShort()
	}

	fun addIndices(count: Int) {
		when (primitiveType) {
			PrimitiveType.SPRITES -> this.addIndicesSprites(count)
			else -> this.addIndicesNormal(count)
		}
	}

	fun addIndicesSprites(count: Int) {
		val OPTIMIZED = false
		//val OPTIMIZED = true

		var maxIdx = 0
		val nsprites = count / 2
		val ivertices = count

		//println("addIndices: size=$size, count=$count")
		when (vertexType.index) {
			IndexEnum.VOID -> {
				// 0..3
				// 2..1
				var bp = this.indexBufferPos
				val start = vertexCount
				if (OPTIMIZED) {
					val end = vertexCount + nsprites * 2
					for (n in 0 until nsprites) {
						indexBuffer[indexBufferPos++] = (start + n * 2 + 0).toShort()
						indexBuffer[indexBufferPos++] = (end + n * 2 + 1).toShort()
						indexBuffer[indexBufferPos++] = (end + n * 2 + 0).toShort()
						indexBuffer[indexBufferPos++] = (end + n * 2 + 0).toShort()
						indexBuffer[indexBufferPos++] = (end + n * 2 + 1).toShort()
						indexBuffer[indexBufferPos++] = (start + n * 2 + 1).toShort()
					}
				} else {
					for (n in 0 until nsprites) {
						indexBuffer[bp++] = (start + n * 4 + 0).toShort()
						indexBuffer[bp++] = (start + n * 4 + 3).toShort()
						indexBuffer[bp++] = (start + n * 4 + 2).toShort()
						indexBuffer[bp++] = (start + n * 4 + 2).toShort()
						indexBuffer[bp++] = (start + n * 4 + 3).toShort()
						indexBuffer[bp++] = (start + n * 4 + 1).toShort()
					}
				}
				this.indexBufferPos = bp

				maxIdx = count
			}
			else -> TODO("addIndicesSprites: ${vertexType.index}, $count")
		}

		// Vertices
		if (OPTIMIZED) {
			val startAddress = 0
			mem.read(state.vertexAddress, vertexBuffer, vertexBufferPos, vertexSize * ivertices)
			vertexBufferPos += vertexSize * ivertices
			state.vertexAddress += vertexSize * ivertices
			vertexCount += ivertices

			val targetAddress = 0
			mem.read(state.vertexAddress, vertexBuffer, vertexBufferPos, vertexSize * ivertices)

			if (vertexType.hasPosition) {
				val posOffsetX = vertexType.posOffset + vertexType.pos.nbytes * 0
				val posOffsetY = vertexType.posOffset + vertexType.pos.nbytes * 1
				val posSize = vertexType.pos.nbytes
				for (n in 0 until nsprites) {
					val offsetX = (n * vertexSize) + posOffsetX
					val offsetY = (n * vertexSize) + posOffsetY

					vertexBuffer.copyRangeTo(startAddress + offsetX, vertexBuffer, targetAddress + offsetX + (vertexSize * 0), posSize)
					vertexBuffer.copyRangeTo(startAddress + offsetY, vertexBuffer, targetAddress + offsetY + (vertexSize * 0), posSize)
					vertexBuffer.copyRangeTo(startAddress + offsetX, vertexBuffer, targetAddress + offsetX + (vertexSize * 1), posSize)
					vertexBuffer.copyRangeTo(startAddress + offsetY, vertexBuffer, targetAddress + offsetY + (vertexSize * 1), posSize)
				}
			}

			vertexBufferPos += vertexSize * ivertices
			vertexCount += ivertices
		} else {
			var vaddr = state.vertexAddress

			val posOffset = vertexType.posOffset
			val posSize = vertexType.pos.nbytes
			val texOffset = vertexType.texOffset
			val texSize = vertexType.tex.nbytes

			var vpos = vertexBufferPos
			for (n in 0 until nsprites) {
				val TLpos = vpos
				val BRpos = vpos + vertexSize

				mem.read(vaddr, vertexBuffer, vpos, vertexSize * 2)
				vpos += vertexSize * 2
				vaddr += vertexSize * 2

				putGenVertex(vpos + vertexSize * 0, TLpos, BRpos, false, true, posOffset, posSize, texOffset, texSize)
				putGenVertex(vpos + vertexSize * 1, TLpos, BRpos, true, false, posOffset, posSize, texOffset, texSize)
				vpos += vertexSize * 2
			}
			this.vertexBufferPos = vpos
			vertexCount += nsprites * 4
			state.vertexAddress = vaddr
		}
	}

	private fun putGenVertex(vertexBufferPos: Int, TLpos: Int, BRpos: Int, gx: Boolean, gy: Boolean, posOffset: Int, posSize: Int, texOffset: Int, texSize: Int) {
		vertexBuffer.copyRangeTo(BRpos, vertexBuffer, vertexBufferPos, vertexSize) // Copy one full

		if (vertexType.hasPosition) {
			vertexBuffer.copyRangeTo((if (!gx) TLpos else BRpos) + posOffset + posSize * 0, vertexBuffer, vertexBufferPos + posOffset + posSize * 0, posSize)
			vertexBuffer.copyRangeTo((if (!gy) TLpos else BRpos) + posOffset + posSize * 1, vertexBuffer, vertexBufferPos + posOffset + posSize * 1, posSize)
		}

		if (vertexType.hasTexture) {
			vertexBuffer.copyRangeTo((if (!gx) TLpos else BRpos) + texOffset + texSize * 0, vertexBuffer, vertexBufferPos + texOffset + texSize * 0, texSize)
			vertexBuffer.copyRangeTo((if (!gy) TLpos else BRpos) + texOffset + texSize * 1, vertexBuffer, vertexBufferPos + texOffset + texSize * 1, texSize)
		}

		// Copy color
		//vertexBuffer.copyRangeTo(BRpos + vertexType.colOffset, vertexBuffer, TLpos + vertexType.colOffset, vertexType.col.nbytes)

	}

	fun addIndicesNormal(count: Int) {
		var maxIdx = 0

		//println("addIndices: size=$size, count=$count")
		when (vertexType.index) {
			IndexEnum.VOID -> {
				val vertexCount = vertexCount
				for (n in 0 until count) indexBuffer[indexBufferPos + n] = (vertexCount + n).toShort()
				indexBufferPos += count
				maxIdx = count
			}
			IndexEnum.SHORT -> {
				val iaddr = state.indexAddress
				for (n in 0 until count) {
					val idx = mem.lhu(iaddr + n * 2)
					maxIdx = max(maxIdx, idx + 1)
					putIndex(idx)
				}
				//println("maxIdx: $maxIdx")
				//state.indexAddress += count * 2
			}
			else -> TODO("addIndices: ${vertexType.index}, $count")
		}

		// Vertices
		mem.read(state.vertexAddress, vertexBuffer, vertexBufferPos, vertexSize * maxIdx)
		vertexBufferPos += vertexSize * maxIdx
		state.vertexAddress += vertexSize * maxIdx
		vertexCount += maxIdx

	}
}
