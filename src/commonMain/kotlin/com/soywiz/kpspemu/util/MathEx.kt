package com.soywiz.kpspemu.util

import kotlin.math.*

fun cosv1(value: Float): Float = cos(value * PI * 0.5).toFloat()
fun sinv1(value: Float): Float {
    var angle = value
    angle -= floor((angle * 0.25f).toDouble()).toFloat() * 4f
    // Handling of specific values first to avoid precision loss in float value
    return when (angle) {
        0f -> 0f
        2f -> -0f
        1f -> 1f
        3f -> -1f
        else -> sin((PI / 2 * angle)).toFloat()
    }
}

fun nsinv1(value: Float): Float = (-sin(value * PI * 0.5)).toFloat()
fun asinv1(value: Float): Float = (asin(value) / (PI * 0.5)).toFloat()

fun Float.clampf(min: Float, max: Float) = when {
    this < min -> min
    this > max -> max
    else -> this
}

fun Float.isNanOrInfinitef() = this.isNaN() || this.isInfinite()

fun scalab(f: Float, scaleFactor: Int): Float = f * 2f.pow(scaleFactor)
