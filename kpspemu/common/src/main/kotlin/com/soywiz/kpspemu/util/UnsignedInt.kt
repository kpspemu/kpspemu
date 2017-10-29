package com.soywiz.kpspemu.util

import com.soywiz.korio.util.IntEx
import com.soywiz.korio.util.toStringUnsigned

data class UnsignedInt(val value: Int) {
	operator fun plus(that: Int) = UnsignedInt(this.value + that)
	operator fun plus(that: UnsignedInt) = UnsignedInt(this.value + that.value)

	operator fun times(that: Int) = UnsignedInt(this.value + that)
	operator fun times(that: UnsignedInt) = UnsignedInt(this.value + that.value)

	override fun toString(): String = value.toStringUnsigned(10)
}

infix fun Int.ult(that: Int) = IntEx.compareUnsigned(this, that) < 0
infix fun Int.ule(that: Int) = IntEx.compareUnsigned(this, that) <= 0
infix fun Int.ugt(that: Int) = IntEx.compareUnsigned(this, that) > 0
infix fun Int.uge(that: Int) = IntEx.compareUnsigned(this, that) >= 0