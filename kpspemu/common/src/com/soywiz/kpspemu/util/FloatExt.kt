package com.soywiz.kpspemu.util

inline fun <T> Iterable<T>.sumByFloat(crossinline selector: (T) -> Float): Float {
    var out = 0f
    for (i in this) out += selector(i)
    return out
}

val Float.pspSign: Float get() = if (this == 0f || this == -0f) 0f else if ((this.toRawBits() ushr 31) != 0) -1f else +1f
val Float.pspAbs: Float get() = Float.fromBits(this.toRawBits() and 0x80000000.toInt().inv())
val Float.pspSat0: Float get() = if (this == -0f) 0f else this.clampf(0f, 1f)
val Float.pspSat1: Float get() = this.clampf(-1f, 1f)

infix fun Float.pspAdd(that: Float): Float = if (this.isNaN() || that.isNaN()) Float.NaN else this + that
infix fun Float.pspSub(that: Float): Float = if (this.isNaN() || that.isNaN()) Float.NaN else this - that
