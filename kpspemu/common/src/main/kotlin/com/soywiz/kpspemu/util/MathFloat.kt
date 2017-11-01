package com.soywiz.kpspemu.util

import kotlin.math.abs
import kotlin.math.pow
import kotlin.math.sign

object MathFloat {
	fun handleCastInfinite(value: Float): Int {
		return if (value < 0) -2147483648 else 2147483647
	}

	fun rintDouble(value: Double): Double {
		val twoToThe52 = 2.0.pow(52); // 2^52
		val sign = sign(value); // preserve sign info
		var rvalue = abs(value);
		if (rvalue < twoToThe52) rvalue = ((twoToThe52 + rvalue) - twoToThe52);
		return sign * rvalue; // restore original sign
	}

	fun rint(value: Float): Int {
		if (value.isNanOrInfinite()) return handleCastInfinite(value)
		return rintDouble(value.toDouble()).toInt()

	}

	fun cast(value: Float): Int {
		if (value.isNanOrInfinite()) return handleCastInfinite(value)
		return if (value < 0) kotlin.math.ceil(value).toInt() else kotlin.math.floor(value).toInt()
	}

	fun trunc(value: Float): Int {
		if (value.isNanOrInfinite()) return handleCastInfinite(value)
		if (value < 0) {
			return kotlin.math.ceil(value).toInt()
		} else {
			return kotlin.math.floor(value).toInt()
		}
	}

	fun round(value: Float): Int {
		if (value.isNanOrInfinite()) return handleCastInfinite(value)
		return kotlin.math.round(value).toInt()
	}

	fun floor(value: Float): Int {
		if (value.isNanOrInfinite()) return handleCastInfinite(value)
		return kotlin.math.floor(value).toInt()
	}

	fun ceil(value: Float): Int {
		if (value.isNanOrInfinite()) return handleCastInfinite(value)
		return kotlin.math.ceil(value).toInt()
	}

	fun isAlmostZero(v: Float): Boolean = abs(v) <= 1e-19
}