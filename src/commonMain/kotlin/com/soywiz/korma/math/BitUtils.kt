package com.soywiz.korma.math

import com.soywiz.kmem.*
import kotlin.jvm.*

object BitUtils {
    @JvmStatic fun bitrev32(rt: Int): Int = rt.reverseBits()
    @JvmStatic fun rotr(rt: Int, pos: Int): Int = rt.rotateRight(pos)
    @JvmStatic fun clz(v: Int): Int = v.countLeadingZeros()
    @JvmStatic fun clo(v: Int): Int = v.countLeadingOnes()
    @JvmStatic fun seb(v: Int): Int = (v shl 24) shr 24
    @JvmStatic fun seh(v: Int): Int = (v shl 16) shr 16
    @JvmStatic fun wsbh(v: Int): Int = ((v and 0xFF00FF00.toInt()) ushr 8) or ((v and 0x00FF00FF) shl 8)
    @JvmStatic fun wsbw(v: Int): Int = ((v and 0xFF000000.toInt()) ushr 24) or
        ((v and 0x00FF0000) ushr 8) or
        ((v and 0x0000FF00) shl 8) or
        ((v and 0x000000FF) shl 24)
}

infix fun Int.ult(that: Int): Boolean = this.toUInt() < that.toUInt()
infix fun Int.udiv(that: Int): Int = (this.toUInt() / that.toUInt()).toInt()
infix fun Int.urem(that: Int): Int = (this.toUInt() % that.toUInt()).toInt()
