package com.soywiz.kpspemu.ge

import com.soywiz.korio.typedarray.copyRangeTo
import com.soywiz.korio.util.extract

class GeState {
	companion object {
		const val STATE_NWORDS = 512
	}

	val data: IntArray = IntArray(STATE_NWORDS)

	val baseAddress: Int get() = (data[Op.BASE] shl 8) and 0xFF000000.toInt()

	var baseOffset: Int
		set(value) = run { data[Op.OFFSETADDR] = (data[Op.OFFSETADDR] and 0xFF000000.toInt()) or ((value ushr 8) and 0x00FFFFFF) }
		get() = data[Op.OFFSETADDR] shl 8

	fun writeInt(key: Int, offset: Int, value: Int): Unit = run { data[offset + data[key]] = value }

	fun setTo(other: GeState) {
		other.data.copyRangeTo(0, this.data, 0, STATE_NWORDS)
	}

	fun clone() = GeState().apply { setTo(this@GeState) }

	// Vertex
	val vertexType: Int get() = data[Op.VERTEXTYPE]
	val vertexReverseNormal: Boolean get() = data[Op.REVERSENORMAL] != 0
	val vertexAddress: Int get() = data[Op.VADDR]
	val vertexTypeTexture: Int get() = VertexType.texture(vertexType)
	val vertexTypeColor: Int get() = VertexType.color(vertexType)
	val vertexTypeNormal: Int get() = VertexType.normal(vertexType)
	val vertexTypePosition: Int get() = VertexType.position(vertexType)
	val vertexTypeWeight: Int get() = VertexType.weight(vertexType)
	val vertexTypeIndex: Int get() = VertexType.index(vertexType)
	val vertexTypeWeightCount: Int get() = VertexType.weightCount(vertexType)
	val vertexTypeMorphCount: Int get() = VertexType.morphingVertexCount(vertexType)
	val vertexTransform2D: Boolean get() = VertexType.transform2D(vertexType)
}

object VertexType {
	fun texture(v: Int) = v.extract(0, 2)
	fun color(v: Int) = v.extract(2, 3)
	fun normal(v: Int) = v.extract(5, 2)
	fun position(v: Int) = v.extract(7, 2)
	fun weight(v: Int) = v.extract(9, 2)
	fun index(v: Int) = v.extract(11, 2)
	fun weightCount(v: Int) = v.extract(14, 3)
	fun morphingVertexCount(v: Int) = v.extract(18, 2)
	fun transform2D(v: Int) = v.extract(23, 1) != 0
}

