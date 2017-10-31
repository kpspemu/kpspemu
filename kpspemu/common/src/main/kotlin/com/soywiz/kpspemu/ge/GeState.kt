package com.soywiz.kpspemu.ge

import com.soywiz.korag.geom.Matrix4
import com.soywiz.korio.typedarray.copyRangeTo
import com.soywiz.korio.util.extract
import com.soywiz.kpspemu.util.safeNextAlignedTo

class GeState {
	companion object {
		const val STATE_NWORDS = 512
	}

	val data: IntArray = IntArray(STATE_NWORDS)

	val baseAddress: Int get() = (data[Op.BASE] shl 8) and 0xFF000000.toInt()

	var baseOffset: Int
		set(value) = run { data[Op.OFFSETADDR] = (data[Op.OFFSETADDR] and 0xFF000000.toInt()) or ((value ushr 8) and 0x00FFFFFF) }
		get() = data[Op.OFFSETADDR] shl 8

	fun writeInt(key: Int, offset: Int, value: Int): Unit = run { data[offset + data[key]++] = value }

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

	fun getAddressRelativeToBaseOffset(address: Int) = (baseAddress or address) + baseOffset
	fun setAddressRelativeToBaseOffset(address: Int) = (address and 0x00FFFFFF) - baseOffset
	fun getProjMatrix(out: Matrix4 = Matrix4()) = out.apply { getMatrix4x4(Op.MAT_PROJ, out) }
	fun getViewMatrix(out: Matrix4 = Matrix4()) = out.apply { getMatrix4x3(Op.MAT_VIEW, out) }
	fun getWorldMatrix(out: Matrix4 = Matrix4()) = out.apply { getMatrix4x3(Op.MAT_WORLD, out) }

	private fun getFloat(key: Int) = Float.fromBits(data[key] shl 8)

	fun getMatrix4x4(register: Int, out: Matrix4) {
		for (n in 0 until 16) {
			out.data[n] = getFloat(register + n)
		}
	}

	fun getMatrix4x3(register: Int, out: Matrix4) {
		var m = 0
		var n = 0
		for (y in 0 until 4) {
			for (x in 0 until 3) {
				out.data[n + x] = getFloat(register + m)
				m++
			}
			n += 4
		}
		out.data[3] = 0f
		out.data[7] = 0f
		out.data[11] = 0f
		out.data[15] = 1f
	}
}

class VertexType(var v: Int = 0) {
	fun init(state: GeState) = this.apply {
		v = state.vertexType
	}

	enum class Attribute(val index: Int) { COLOR(0), NORMAL(1), POSITION(2), TEXTURE(3), WEIGHTS(4), END(5) }

	val texture: NumericEnum get() = NumericEnum(v.extract(0, 2))
	val color: ColorEnum get() = ColorEnum(v.extract(2, 3))
	val normal: NumericEnum get() = NumericEnum(v.extract(5, 2))
	val position: NumericEnum get() = NumericEnum(v.extract(7, 2))
	val weight: NumericEnum get() = NumericEnum(v.extract(9, 2))
	val index: IndexEnum get() = IndexEnum(v.extract(11, 2))
	val weightCount: Int get() = v.extract(14, 3)
	val morphingVertexCount: Int get() = v.extract(18, 2)
	val transform2D: Boolean get() = v.extract(23, 1) != 0

	val hasIndices: Boolean get() = index != IndexEnum.VOID
	val hasTexture: Boolean get() = texture != NumericEnum.VOID
	val hasColor: Boolean get() = color != ColorEnum.VOID
	val hasNormal: Boolean get() = normal != NumericEnum.VOID
	val hasPosition: Boolean get() = position != NumericEnum.VOID
	val hasWeight: Boolean get() = weight != NumericEnum.VOID

	fun offsetOf(attribute: Attribute): Int {
		var out = 0
		val color = color
		val components = if (transform2D) 2 else 3
		val normal = normal
		val position = position
		val weight = weight
		val weightCount = weightCount
		val texture = texture

		out = out.safeNextAlignedTo(color.nbytes)
		if (attribute == Attribute.COLOR) return out
		out += color.nbytes

		out = out.safeNextAlignedTo(normal.nbytes)
		if (attribute == Attribute.NORMAL) return out
		out += normal.nbytes * components

		out = out.safeNextAlignedTo(position.nbytes)
		if (attribute == Attribute.POSITION) return out
		out += position.nbytes * components

		out = out.safeNextAlignedTo(texture.nbytes)
		if (attribute == Attribute.TEXTURE) return out
		out += texture.nbytes * 2

		out = out.safeNextAlignedTo(weight.nbytes)
		if (attribute == Attribute.WEIGHTS) return out
		out += weight.nbytes * weightCount

		return out
	}

	fun size(): Int = offsetOf(Attribute.END)
}

enum class IndexEnum(val id: Int, val nbytes: Int) {
	VOID(0, 0),
	BYTE(1, 1),
	SHORT(2, 2);

	companion object {
		val values = values()
		operator fun invoke(id: Int) = values[id]
	}
}

enum class NumericEnum(val id: Int, val nbytes: Int) {
	VOID(0, 0),
	BYTE(1, 1),
	SHORT(2, 2),
	FLOAT(3, 4);

	companion object {
		val values = values()
		operator fun invoke(id: Int) = values[id]
	}
}

enum class ColorEnum(val id: Int, val nbytes: Int) {
	VOID(0, 0),
	INVALID1(1, 0),
	INVALID2(2, 0),
	INVALID3(3, 0),
	COLOR5650(4, 2),
	COLOR5551(5, 2),
	COLOR4444(6, 2),
	COLOR8888(7, 4);

	companion object {
		val values = values()
		operator fun invoke(id: Int) = values[id]
	}
}

enum class PrimitiveType(val id: Int) {
	POINTS(0),
	LINES(1),
	LINE_STRIP(2),
	TRIANGLES(3),
	TRIANGLE_STRIP(4),
	TRIANGLE_FAN(5),
	SPRITES(6);

	companion object {
		val values = values()
		operator fun invoke(id: Int): PrimitiveType = values[id]
	}
}

