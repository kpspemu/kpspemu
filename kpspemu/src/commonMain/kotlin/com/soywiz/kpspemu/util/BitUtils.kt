package com.soywiz.kpspemu.util

import com.soywiz.kmem.*

object BitUtils {
    fun mask(value: Int): Int = value.mask()
    fun bitrev32(x: Int): Int = x.reverseBits()
    fun rotr(value: Int, offset: Int): Int = value.rotateRight(offset)
    fun clz32(x: Int): Int = x.countLeadingZeros()
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
