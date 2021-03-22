package com.soywiz.kpspemu.util

import com.soywiz.kmem.*
import kotlin.math.*

object Math {
    private fun handleCastInfinite(value: Double): Int = if (value < 0) -2147483648 else 2147483647
    fun rintDouble(value: Double): Double {
        val twoToThe52 = 2.0.pow(52) // 2^52
        val sign = kotlin.math.sign(value) // preserve sign info
        var rvalue = kotlin.math.abs(value)
        if (rvalue < twoToThe52) rvalue = ((twoToThe52 + rvalue) - twoToThe52)
        return sign * rvalue // restore original sign
    }

    fun rintChecked(value: Double): Int {
        if (value.isNanOrInfinite()) return handleCastInfinite(value)
        return rintDouble(value.toDouble()).toInt()
    }

    fun castChecked(value: Double): Int {
        if (value.isNanOrInfinite()) return handleCastInfinite(value)
        return if (value < 0) kotlin.math.ceil(value).toInt() else kotlin.math.floor(value).toInt()
    }

    fun truncChecked(value: Double): Int {
        if (value.isNanOrInfinite()) return handleCastInfinite(value)
        return if (value < 0) kotlin.math.ceil(value).toInt() else kotlin.math.floor(value).toInt()
    }

    fun roundChecked(value: Double): Int {
        if (value.isNanOrInfinite()) return handleCastInfinite(value)
        return kotlin.math.round(value).toInt()
    }

    fun floorChecked(value: Double): Int {
        if (value.isNanOrInfinite()) return handleCastInfinite(value)
        return kotlin.math.floor(value).toInt()
    }

    fun ceilChecked(value: Double): Int {
        if (value.isNanOrInfinite()) return handleCastInfinite(value)
        return kotlin.math.ceil(value).toInt()
    }

    fun rint(v: Float): Int = rintChecked(v.toDouble())
    fun cast(v: Float): Int = castChecked(v.toDouble())
    fun ceil(v: Float): Int = ceilChecked(v.toDouble())
    fun floor(v: Float): Int = floorChecked(v.toDouble())
    fun trunc(v: Float): Int = truncChecked(v.toDouble())
    fun round(v: Float): Int = roundChecked(v.toDouble())

    /*
    fun rint(v: Float): Int = if (v >= floor(v) + 0.5) ceil(v) else round(v)
    fun cast(fs: Float): Int = fs.toInt()
    fun ceil(v: Float): Int = v.toIntCeil()
    fun floor(v: Float): Int = v.toIntFloor()
    fun trunc(v: Float): Int = v.toIntFloor()
    fun round(v: Float): Int = v.toIntRound()
     */
}
