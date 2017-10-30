package com.soywiz.kpspemu.util

import kotlin.math.abs

fun imul32_64(a: Int, b: Int, result: IntArray = IntArray(2)): IntArray {
	if (a == 0) {
		result[0] = 0
		result[1] = 0;
		return result;
	}
	if (b == 0) {
		result[0] = 0
		result[1] = 0;
		return result;
	}

	if ((a >= -32768 && a <= 32767) && (b >= -32768 && b <= 32767)) {
		result[0] = a * b;
		result[1] = if (result[0] < 0) -1 else 0;
		return result;
	}

	val doNegate = (a < 0) xor (b < 0);

	umul32_64(abs(a), abs(b), result);

	if (doNegate) {
		result[0] = result [0].inv();
		result[1] = result [1].inv();
		result[0] = (result[0] + 1) or 0;
		if (result[0] == 0) result[1] = (result[1] + 1) or 0;
	}

	return result;
}

fun umul32_64(a: Int, b: Int, result: IntArray = IntArray(2)): IntArray {
	if (a ult  32767 && b ult  65536) {
		result[0] = a * b;
		result[1] = if (result[0] < 0) -1 else 0;
		return result;
	}

	val a00 = a and 0xFFFF
	val a16 = a ushr 16;
	val b00 = b and 0xFFFF
	val b16 = b ushr 16;
	val c00 = a00 * b00;
	var c16 = (c00 ushr 16) + (a16 * b00);
	var c32 = c16 ushr 16;
	c16 = (c16 and 0xFFFF) + (a00 * b16);
	c32 += c16 ushr 16;
	var c48 = c32 ushr 16;
	c32 = (c32 and 0xFFFF) + (a16 * b16);
	c48 += c32 ushr 16;

	result[0] = ((c16 and 0xFFFF) shl 16) or (c00 and 0xFFFF);
	result[1] = ((c48 and 0xFFFF) shl 16) or (c32 and 0xFFFF);
	return result;
};