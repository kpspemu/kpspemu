package com.soywiz.kpspemu.util

import com.soywiz.korio.util.IntEx
import kotlin.math.abs

fun imul32_64(a: Int, b: Int, result: IntArray = IntArray(2)): IntArray {
	if (a == 0) {
		result[0] = 0
		result[1] = 0
		return result
	}
	if (b == 0) {
		result[0] = 0
		result[1] = 0
		return result
	}

	if ((a >= -32768 && a <= 32767) && (b >= -32768 && b <= 32767)) {
		result[0] = a * b
		result[1] = if (result[0] < 0) -1 else 0
		return result
	}

	val doNegate = (a < 0) xor (b < 0)

	umul32_64(abs(a), abs(b), result)

	if (doNegate) {
		result[0] = result[0].inv()
		result[1] = result[1].inv()
		result[0] = (result[0] + 1) or 0
		if (result[0] == 0) result[1] = (result[1] + 1) or 0
	}

	return result
}

fun umul32_64(a: Int, b: Int, result: IntArray = IntArray(2)): IntArray {
	if (a ult 32767 && b ult 65536) {
		result[0] = a * b
		result[1] = if (result[0] < 0) -1 else 0
		return result
	}

	val a00 = a and 0xFFFF
	val a16 = a ushr 16
	val b00 = b and 0xFFFF
	val b16 = b ushr 16
	val c00 = a00 * b00
	var c16 = (c00 ushr 16) + (a16 * b00)
	var c32 = c16 ushr 16
	c16 = (c16 and 0xFFFF) + (a00 * b16)
	c32 += c16 ushr 16
	var c48 = c32 ushr 16
	c32 = (c32 and 0xFFFF) + (a16 * b16)
	c48 += c32 ushr 16

	result[0] = ((c16 and 0xFFFF) shl 16) or (c00 and 0xFFFF)
	result[1] = ((c48 and 0xFFFF) shl 16) or (c32 and 0xFFFF)
	return result
}

val Int.unsigned: Long get() = this.toLong() and 0xFFFFFFFF

object BitUtils {
	fun mask(value: Int): Int = (1 shl value) - 1
	fun bitrev32(x: Int): Int {
		var v = x
		v = ((v ushr 1) and 0x55555555) or ((v and 0x55555555) shl 1); // swap odd and even bits
		v = ((v ushr 2) and 0x33333333) or ((v and 0x33333333) shl 2); // swap consecutive pairs
		v = ((v ushr 4) and 0x0F0F0F0F) or ((v and 0x0F0F0F0F) shl 4); // swap nibbles ... 
		v = ((v ushr 8) and 0x00FF00FF) or ((v and 0x00FF00FF) shl 8); // swap bytes
		v = ((v ushr 16) and 0x0000FFFF) or ((v and 0x0000FFFF) shl 16); // swap 2-byte long pairs
		return v;
	}

	fun rotr(value: Int, offset: Int): Int = (value ushr offset) or (value shl (32 - offset))

	fun clz32(x: Int): Int {
		var v = x
		if (v == 0) return 32;
		var result = 0;
		// Binary search.
		if ((v and 0xFFFF0000.toInt()) == 0) run { v = v shl 16; result += 16; }
		if ((v and 0xFF000000.toInt()) == 0) run { v = v shl 8; result += 8; }
		if ((v and 0xF0000000.toInt()) == 0) run { v = v shl 4; result += 4; }
		if ((v and 0xC0000000.toInt()) == 0) run { v = v shl 2; result += 2; }
		if ((v and 0x80000000.toInt()) == 0) run { v = v shl 1; result += 1; }
		return result;
	}

	fun clo(x: Int): Int = clz32(x.inv())
	fun clz(x: Int): Int = clz32(x)
	fun seb(x: Int): Int = (x shl 24) shr 24
	fun seh(x: Int): Int = (x shl 16) shr 16

	fun wsbh(v: Int): Int = ((v and 0xFF00FF00.toInt()) ushr 8) or ((v and 0x00FF00FF) shl 8)

	fun wsbw(v: Int): Int = (
		((v and 0xFF000000.toInt()) ushr 24) or
			((v and 0x00FF0000) ushr 8) or
			((v and 0x0000FF00) shl 8) or
			((v and 0x000000FF) shl 24)
		)
}

fun Int.compareToUnsigned(that: Int) = IntEx.compareUnsigned(this, that)

fun Int.safeNextAlignedTo(align: Int) = when {
	(align == 0) || (this % align == 0) -> this
	else -> (((this / align) + 1) * align)
}

infix fun Int.ult(that: Int) = IntEx.compareUnsigned(this, that) < 0
infix fun Int.ule(that: Int) = IntEx.compareUnsigned(this, that) <= 0
infix fun Int.ugt(that: Int) = IntEx.compareUnsigned(this, that) > 0
infix fun Int.uge(that: Int) = IntEx.compareUnsigned(this, that) >= 0
