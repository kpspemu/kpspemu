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
	var vertexAddress: Int
		set(value) = run { data[Op.VADDR] = setAddressRelativeToBaseOffset(data[Op.VADDR]) }
		get() = getAddressRelativeToBaseOffset(data[Op.VADDR])
	val indexAddress: Int get() = getAddressRelativeToBaseOffset(data[Op.IADDR])
	//val vertexAddress: Int get() = data[Op.VADDR]
	val vertexTypeTexture: Int get() = VertexType.texture(vertexType)
	val vertexTypeColor: Int get() = VertexType.color(vertexType)
	val vertexTypeNormal: Int get() = VertexType.normal(vertexType)
	val vertexTypePosition: Int get() = VertexType.position(vertexType)
	val vertexTypeWeight: Int get() = VertexType.weight(vertexType)
	val vertexTypeIndex: Int get() = VertexType.index(vertexType)
	val vertexTypeWeightCount: Int get() = VertexType.weightCount(vertexType)
	val vertexTypeMorphCount: Int get() = VertexType.morphingVertexCount(vertexType)
	val vertexTransform2D: Boolean get() = VertexType.transform2D(vertexType)
	val vertexSize: Int get() = VertexType.size(vertexType)

	fun getAddressRelativeToBaseOffset(address: Int) = (baseAddress or address) + baseOffset
	fun setAddressRelativeToBaseOffset(address: Int) = (address and 0x00FFFFFF) - baseOffset
}

object VertexType {
	val COLOR_SIZES = intArrayOf(0, 0, 0, 0, 2, 2, 2, 4)
	val INDEX_SIZES = intArrayOf(0, 1, 2)
	val NUMERIC_SIZES = intArrayOf(0, 1, 2, 4)

	fun texture(v: Int) = v.extract(0, 2)
	fun color(v: Int) = v.extract(2, 3)
	fun normal(v: Int) = v.extract(5, 2)
	fun position(v: Int) = v.extract(7, 2)
	fun weight(v: Int) = v.extract(9, 2)
	fun index(v: Int) = v.extract(11, 2)
	fun weightCount(v: Int) = v.extract(14, 3)
	fun morphingVertexCount(v: Int) = v.extract(18, 2)
	fun transform2D(v: Int) = v.extract(23, 1) != 0

	fun size(v: Int): Int {
		var out = 0
		val components = if (transform2D(v)) 2 else 3
		out += COLOR_SIZES[color(v)]
		out += NUMERIC_SIZES[normal(v)] * components
		out += NUMERIC_SIZES[position(v)] * components
		out += NUMERIC_SIZES[weight(v)] * components * weightCount(v)
		return out
	}
}

object IndexEnum {
	val Void = 0
	val Byte = 1
	val Short = 2
}

object NumericEnum {
	val Void = 0
	val Byte = 1
	val Short = 2
	val Float = 3
}

object ColorEnum {
	val Void = 0
	val Invalid1 = 1
	val Invalid2 = 2
	val Invalid3 = 3
	val Color5650 = 4
	val Color5551 = 5
	val Color4444 = 6
	val Color8888 = 7
}
