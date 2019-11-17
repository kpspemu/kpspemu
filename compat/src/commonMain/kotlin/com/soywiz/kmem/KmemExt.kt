package com.soywiz.kmem

import com.soywiz.korio.util.encoding.*

typealias KmlNativeBuffer = MemBuffer

val MemBuffer.i32 get() = this.asInt32Buffer()

val UByteArray.data get() = this.toByteArray()
fun UByteArray(data: ByteArray) = data.toUByteArray()
infix fun Int.hasFlag(v: Int): Boolean = (this and v) == v
