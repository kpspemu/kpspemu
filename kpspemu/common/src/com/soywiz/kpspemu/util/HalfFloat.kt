package com.soywiz.kpspemu.util

import com.soywiz.kmem.*

// Move this outside
fun HalfFloat.toFloat() = this.f

fun Float.toHalfFloat() = HalfFloat(this)
data class HalfFloat(val v: Char) {
    constructor(v: Float) : this(floatBitsToHalfFloatBits(v.reinterpretAsInt()).toChar())

    val f: Float get() = halfFloatBitsToFloatBits(v.toInt()).reinterpretAsFloat()
    override fun toString(): String = "$f"
    fun toBits() = v

    companion object {
        fun fromBits(v: Char) = HalfFloat(v)

        fun halfFloatBitsToFloat(imm16: Int): Float = Float.fromBits(halfFloatBitsToFloatBits(imm16))
        fun floatToHalfFloatBits(i: Float): Int = floatBitsToHalfFloatBits(i.toRawBits())

        fun halfFloatBitsToFloatBits(imm16: Int): Int {
            val s = imm16 shr 15 and 0x00000001 // sign
            var e = imm16 shr 10 and 0x0000001f // exponent
            var f = imm16 shr 0 and 0x000003ff // fraction

            // need to handle 0x7C00 INF and 0xFC00 -INF?
            if (e == 0) {
                // need to handle +-0 case f==0 or f=0x8000?
                if (f == 0) {
                    // Plus or minus zero
                    return s shl 31
                }
                // Denormalized number -- renormalize it
                while (f and 0x00000400 == 0) {
                    f = f shl 1
                    e -= 1
                }
                e += 1
                f = f and 0x00000400.inv()
            } else if (e == 31) {
                return if (f == 0) {
                    // Inf
                    s shl 31 or 0x7f800000
                } else s shl 31 or 0x7f800000 or f
                // NaN
                // fraction is not shifted by PSP
            }

            e += (127 - 15)
            f = f shl 13

            return s shl 31 or (e shl 23) or f
        }

        fun floatBitsToHalfFloatBits(i: Int): Int {
            val s = i shr 16 and 0x00008000              // sign
            val e = (i shr 23 and 0x000000ff) - (127 - 15) // exponent
            var f = i shr 0 and 0x007fffff              // fraction

            // need to handle NaNs and Inf?
            if (e <= 0) {
                if (e < -10) {
                    return if (s != 0) {
                        // handle -0.0
                        0x8000
                    } else 0
                }
                f = f or 0x00800000 shr 1 - e
                return s or (f shr 13)
            } else if (e == 0xff - (127 - 15)) {
                if (f == 0) {
                    // Inf
                    return s or 0x7c00
                }
                // NAN
                f = f shr 13
                f = 0x3ff // PSP always encodes NaN with this value
                return s or 0x7c00 or f or if (f == 0) 1 else 0
            }
            return if (e > 30) {
                // Overflow
                s or 0x7c00
            } else s or (e shl 10) or (f shr 13)
        }
    }
}
