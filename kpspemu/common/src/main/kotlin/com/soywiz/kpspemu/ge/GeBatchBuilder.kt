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
		if (this.primitiveType != primitiveType || this.vertexType.v != state.vertexType) {
			flush()
		}
		vertexType.init(state)
		this.primitiveType = primitiveType
		this.vertexSize = vertexType.size()
	}

	fun tflush() = Unit
	fun tsync() = Unit

	fun flush() {
		//println("flush: $indexBufferPos")
		if (indexBufferPos > 0) {
			ge.emitBatch(GeBatch(ge.state.clone(), primitiveType ?: PrimitiveType.TRIANGLES, indexBufferPos, vertexBuffer.copyOf(vertexBufferPos), indexBuffer.copyOf(indexBufferPos)))
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

	private fun putGenVertex(TLpos: Int, BRpos: Int, gx: Boolean, gy: Boolean) {
		vertexBuffer.copyRangeTo(BRpos, vertexBuffer, vertexBufferPos, vertexSize) // Copy one full

		if (vertexType.hasPosition) {
			vertexBuffer.copyRangeTo((if (!gx) TLpos else BRpos) + vertexType.posOffset + vertexType.pos.nbytes * 0, vertexBuffer, vertexBufferPos + vertexType.posOffset + vertexType.pos.nbytes * 0, vertexType.pos.nbytes)
			vertexBuffer.copyRangeTo((if (!gy) TLpos else BRpos) + vertexType.posOffset + vertexType.pos.nbytes * 1, vertexBuffer, vertexBufferPos + vertexType.posOffset + vertexType.pos.nbytes * 1, vertexType.pos.nbytes)
		}

		if (vertexType.hasTexture) {
			vertexBuffer.copyRangeTo((if (!gx) TLpos else BRpos) + vertexType.texOffset + vertexType.tex.nbytes * 0, vertexBuffer, vertexBufferPos + vertexType.texOffset + vertexType.tex.nbytes * 0, vertexType.tex.nbytes)
			vertexBuffer.copyRangeTo((if (!gy) TLpos else BRpos) + vertexType.texOffset + vertexType.tex.nbytes * 1, vertexBuffer, vertexBufferPos + vertexType.texOffset + vertexType.tex.nbytes * 1, vertexType.tex.nbytes)
		}

		// Copy color
		vertexBuffer.copyRangeTo(BRpos + vertexType.colOffset, vertexBuffer, TLpos + vertexType.colOffset, vertexType.col.nbytes)

		vertexBufferPos += vertexSize
		vertexCount++
	}

	fun addIndices(count: Int) {
		var maxIdx = 0

		//println("addIndices: size=$size, count=$count")
		when (vertexType.index) {
			IndexEnum.VOID -> {
				when (primitiveType) {
					PrimitiveType.SPRITES -> {
						var m = vertexCount
						val nsprites = count / 2
						for (n in 0 until nsprites) {
							// 0..3
							// 2..1

							putIndex(m + 0)
							putIndex(m + 3)
							putIndex(m + 2)

							putIndex(m + 2)
							putIndex(m + 3)
							putIndex(m + 1)

							m += 4
						}
					}
					else -> {
						for (n in 0 until count) putIndex(vertexCount + n)
					}
				}
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
		when (primitiveType) {
			PrimitiveType.SPRITES -> {
				var vaddr = state.vertexAddress
				for (n in 0 until maxIdx / 2) {
					val TLpos = vertexBufferPos
					putVertex(vaddr); vaddr += vertexSize // TL
					val BRpos = vertexBufferPos
					putVertex(vaddr); vaddr += vertexSize // BR

					putGenVertex(TLpos, BRpos, false, true)
					putGenVertex(TLpos, BRpos, true, false)
				}
				state.vertexAddress = vaddr
			}
			else -> {
				var vaddr = state.vertexAddress
				for (n in 0 until maxIdx) {
					putVertex(vaddr)
					vaddr += vertexSize
				}
				state.vertexAddress = vaddr
			}
		}
	}
}
