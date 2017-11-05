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
					//val end = start + nsprites * 2
					//for (n in 0 until nsprites) {
					//	indexBuffer[indexBufferPos++] = (start + n * 2 + 0).toShort()
					//	indexBuffer[indexBufferPos++] = (end + n * 2 + 1).toShort()
					//	indexBuffer[indexBufferPos++] = (end + n * 2 + 0).toShort()
					//	indexBuffer[indexBufferPos++] = (end + n * 2 + 0).toShort()
					//	indexBuffer[indexBufferPos++] = (end + n * 2 + 1).toShort()
					//	indexBuffer[indexBufferPos++] = (start + n * 2 + 1).toShort()
					//}
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
		val vertexSize_2 = vertexSize * 2
		if (OPTIMIZED) {
			//val posOffset = vertexType.posOffset
			//val posSize = vertexType.pos.nbytes
			//val texOffset = vertexType.texOffset
			//val texSize = vertexType.tex.nbytes
//
			//var vaddr = state.vertexAddress
			//var vpos = vertexBufferPos
			//var vpos2 = vertexBufferPos + vertexSize_2 * nsprites
			//for (n in 0 until nsprites) {
			//	val TLpos = vpos
			//	val BRpos = vpos + vertexSize
//
			//	mem.read(vaddr, vertexBuffer, vpos, vertexSize * 2)
			//	vpos += vertexSize_2
			//	vaddr += vertexSize_2
//
			//	putGenVertex(vpos2 + vertexSize * 0, TLpos, BRpos, false, true, posOffset, posSize, texOffset, texSize)
			//	putGenVertex(vpos2 + vertexSize * 1, TLpos, BRpos, true, false, posOffset, posSize, texOffset, texSize)
			//	vpos2 += vertexSize_2
			//}
			//this.vertexBufferPos = vpos2
			//vertexCount += nsprites * 4
			//state.vertexAddress = vaddr

		} else {
			val posSize = vertexType.pos.nbytes
			val posOffsetX = vertexType.posOffset
			val posOffsetY = vertexType.posOffset + posSize
			val texSize = vertexType.tex.nbytes
			val texOffsetX = vertexType.texOffset
			val texOffsetY = vertexType.texOffset + texSize

			var vaddr = state.vertexAddress
			var vpos = vertexBufferPos
			for (n in 0 until nsprites) {
				val TLpos = vpos
				val BRpos = vpos + vertexSize

				mem.read(vaddr + (n * vertexSize_2), vertexBuffer, vpos, vertexSize * 2)
				vpos += vertexSize_2

				putGenVertex(vpos + vertexSize * 0, BRpos, TLpos, BRpos, posSize, posOffsetX, posOffsetY, texSize, texOffsetX, texOffsetY)
				putGenVertex(vpos + vertexSize * 1, BRpos, BRpos, TLpos, posSize, posOffsetX, posOffsetY, texSize, texOffsetX, texOffsetY)

				vpos += vertexSize_2
			}
			vertexCount += nsprites * 4
			this.vertexBufferPos = vpos
			state.vertexAddress += nsprites * vertexSize * 2
		}
	}

	private fun putGenVertex(vertexBufferPos: Int, base: Int,  gx: Int, gy: Int, posSize: Int, posOffsetX: Int, posOffsetY: Int, texSize: Int, texOffsetX: Int, texOffsetY: Int) {
		vertexBuffer.copyRangeTo(base, vertexBuffer, vertexBufferPos, vertexSize) // Copy one full

		if (vertexType.hasPosition) {
			vertexBuffer.copyRangeTo(gx + posOffsetX, vertexBuffer, vertexBufferPos + posOffsetX, posSize)
			vertexBuffer.copyRangeTo(gy + posOffsetY, vertexBuffer, vertexBufferPos + posOffsetY, posSize)
		}

		if (vertexType.hasTexture) {
			vertexBuffer.copyRangeTo(gx + texOffsetX, vertexBuffer, vertexBufferPos + texOffsetX, texSize)
			vertexBuffer.copyRangeTo(gy + texOffsetY, vertexBuffer, vertexBufferPos + texOffsetY, texSize)
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
