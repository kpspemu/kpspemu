package com.soywiz.kpspemu.ge

import com.soywiz.korag.geom.Matrix4
import com.soywiz.korim.color.RGBA_4444
import com.soywiz.korim.color.RGBA_5551
import com.soywiz.korim.color.RGB_565
import com.soywiz.korio.stream.*
import com.soywiz.korio.typedarray.copyRangeTo
import com.soywiz.korio.util.extract
import com.soywiz.korio.util.nextAlignedTo
import com.soywiz.kpspemu.util.hex
import kotlin.math.max

// struct PspGeContext { unsigned int context[512] }
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
	fun init(v: Int) = this.apply {
		this.v = v
	}

	enum class Attribute(val index: Int) { WEIGHTS(0), TEXTURE(1), COLOR(2), NORMAL(3), POSITION(4), END(5) }

	val tex: NumericEnum get() = NumericEnum(v.extract(0, 2))
	val col: ColorEnum get() = ColorEnum(v.extract(2, 3))
	val normal: NumericEnum get() = NumericEnum(v.extract(5, 2))
	val pos: NumericEnum get() = NumericEnum(v.extract(7, 2))
	val weight: NumericEnum get() = NumericEnum(v.extract(9, 2))
	val index: IndexEnum get() = IndexEnum(v.extract(11, 2))
	val weightComponents: Int get() = v.extract(14, 3)
	val morphingVertexCount: Int get() = v.extract(18, 2)
	val transform2D: Boolean get() = v.extract(23, 1) != 0

	val hasIndices: Boolean get() = index != IndexEnum.VOID
	val hasTexture: Boolean get() = tex != NumericEnum.VOID
	val hasColor: Boolean get() = col != ColorEnum.VOID
	val hasNormal: Boolean get() = normal != NumericEnum.VOID
	val hasPosition: Boolean get() = pos != NumericEnum.VOID
	val hasWeight: Boolean get() = weight != NumericEnum.VOID

	//val components: Int get() = if (transform2D) 2 else 3
	val components: Int get() = 3

	val posComponents: Int get() = components // @TODO: Verify this
	val normalComponents: Int get() = components // @TODO: Verify this
	val texComponents: Int get() = 2 // @TODO: texture components must be 2 or 3

	val colorSize: Int get() = col.nbytes
	val normalSize: Int get() = normal.nbytes * normalComponents
	val positionSize: Int get() = pos.nbytes * posComponents
	val textureSize: Int get() = tex.nbytes * texComponents
	val weightSize: Int get() = weight.nbytes * weightComponents

	val colOffset: Int get() = offsetOf(Attribute.COLOR)
	val normalOffset: Int get() = offsetOf(Attribute.NORMAL)
	val posOffset: Int get() = offsetOf(Attribute.POSITION)
	val texOffset: Int get() = offsetOf(Attribute.TEXTURE)
	val weightOffset: Int get() = offsetOf(Attribute.WEIGHTS)

	fun offsetOf(attribute: Attribute): Int {
		var out = 0

		out = out.nextAlignedTo(weight.nbytes)
		if (attribute == Attribute.WEIGHTS) return out
		out += weightSize

		out = out.nextAlignedTo(tex.nbytes)
		if (attribute == Attribute.TEXTURE) return out
		out += textureSize

		out = out.nextAlignedTo(col.nbytes)
		if (attribute == Attribute.COLOR) return out
		out += colorSize

		out = out.nextAlignedTo(normal.nbytes)
		if (attribute == Attribute.NORMAL) return out
		out += normalSize

		out = out.nextAlignedTo(pos.nbytes)
		if (attribute == Attribute.POSITION) return out
		out += positionSize

		out = out.nextAlignedTo(max(max(max(max(weight.nbytes, tex.nbytes), col.nbytes), normal.nbytes), pos.nbytes))

		return out
	}

	fun size(): Int = offsetOf(Attribute.END)

	override fun toString(): String {
		val parts = arrayListOf<String>()
		parts += "color=$col"
		parts += "normal=$normal"
		parts += "pos=$pos"
		parts += "tex=$tex"
		parts += "weight=$weight"
		parts += "size=${size()}"
		return "VertexType(${parts.joinToString(", ")})"
	}
}

fun VertexType.init(state: GeState) = init(state.vertexType)

@Suppress("ArrayInDataClass")
data class VertexRaw(
	var color: Int = 0,
	val normal: FloatArray = FloatArray(3),
	val pos: FloatArray = FloatArray(3),
	val tex: FloatArray = FloatArray(3),
	val weights: FloatArray = FloatArray(8)
) {
	override fun toString(): String =
		"VertexRaw(${color.hex}, normal=${normal.toList()}, pos=${pos.toList()}, tex=${tex.toList()}, weights=${weights.toList()})"
}

class VertexReader {
	private fun SyncStream.readBytes(count: Int, out: FloatArray = FloatArray(4), normalized: Boolean) = out.apply {
		skipToAlign(4)
		if (normalized) {
			for (n in 0 until count) out[n] = readS8().toFloat() / 0x7F
		} else {
			for (n in 0 until count) out[n] = readS8().toFloat()
		}
	}

	private fun SyncStream.readShorts(count: Int, out: FloatArray = FloatArray(4), normalized: Boolean) = out.apply {
		skipToAlign(4)
		if (normalized) {
			for (n in 0 until count) out[n] = readS16_le().toFloat() / 0x7FFF
		} else {
			for (n in 0 until count) out[n] = readS16_le().toFloat()
		}
	}

	private fun SyncStream.readFloats(count: Int, out: FloatArray = FloatArray(4), normalized: Boolean) = out.apply {
		skipToAlign(4)
		for (n in 0 until count) out[n] = readF32_le()
	}

	fun SyncStream.readColorType(type: ColorEnum): Int {
		return when (type) {
			ColorEnum.COLOR4444 -> RGBA_4444.packRGBA(readU16_le())
			ColorEnum.COLOR5551 -> RGBA_5551.packRGBA(readU16_le())
			ColorEnum.COLOR5650 -> RGB_565.packRGBA(readU16_le())
			ColorEnum.COLOR8888 -> readS32_le()
			else -> TODO()
		}
	}

	fun SyncStream.readNumericType(count: Int, type: NumericEnum, out: FloatArray = FloatArray(4), normalized: Boolean): FloatArray = out.apply {
		when (type) {
			NumericEnum.VOID -> Unit
			NumericEnum.BYTE -> readBytes(count, out, normalized)
			NumericEnum.SHORT -> readShorts(count, out, normalized)
			NumericEnum.FLOAT -> readFloats(count, out, normalized)
		}
	}

	fun readOne(s: SyncStream, type: VertexType, out: VertexRaw = VertexRaw()): VertexRaw {
		s.safeSkipToAlign(type.weight.nbytes)
		s.readNumericType(type.weightComponents, type.weight, out.weights, normalized = true)

		s.safeSkipToAlign(type.tex.nbytes)
		s.readNumericType(type.texComponents, type.tex, out.tex, normalized = true)

		s.safeSkipToAlign(type.col.nbytes)
		out.color = s.readColorType(type.col)

		s.safeSkipToAlign(type.normal.nbytes)
		s.readNumericType(type.normalComponents, type.normal, out.normal, normalized = false)

		s.safeSkipToAlign(type.pos.nbytes)
		s.readNumericType(type.posComponents, type.pos, out.pos, normalized = false)

		s.safeSkipToAlign(max(max(max(max(type.weight.nbytes, type.tex.nbytes), type.col.nbytes), type.normal.nbytes), type.pos.nbytes))

		return out
	}

	fun read(type: VertexType, count: Int, s: SyncStream) = (0 until count).map { readOne(s, type) }
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

private fun SyncStream.safeSkipToAlign(alignment: Int) = when {
	alignment == 0 -> Unit
	else -> this.skipToAlign(alignment)
}
