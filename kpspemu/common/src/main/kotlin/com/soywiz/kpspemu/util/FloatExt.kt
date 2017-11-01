package com.soywiz.kpspemu.util

fun Float.clamp(min: Float, max: Float) = when {
	(this < min) -> min
	(this > max) -> max
	else -> this
}

fun Float.isNanOrInfinite() = this.isNaN() || this.isInfinite()