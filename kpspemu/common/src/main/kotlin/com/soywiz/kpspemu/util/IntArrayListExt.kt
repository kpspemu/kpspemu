package com.soywiz.kpspemu.util

import com.soywiz.kds.*

fun IntArrayList.copyOfIntArray() = this.data.copyOf(this.size)
fun IntArrayList.copyOfShortArray(): ShortArray {
    val out = ShortArray(this.size)
    for (n in 0 until this.size) out[n] = this.data[n].toShort()
    return out
}
