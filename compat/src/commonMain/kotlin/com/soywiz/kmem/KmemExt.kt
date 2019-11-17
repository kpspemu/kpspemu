package com.soywiz.kmem

import com.soywiz.korio.util.encoding.*

typealias KmlNativeBuffer = MemBuffer

val MemBuffer.i8 get() = this.asInt8Buffer()
val MemBuffer.i16 get() = this.asInt16Buffer()
val MemBuffer.i32 get() = this.asInt32Buffer()
val MemBuffer.f32 get() = this.asFloat32Buffer()

@UseExperimental(kotlin.ExperimentalUnsignedTypes::class)
val UByteArray.data get() = this.toByteArray()
@UseExperimental(kotlin.ExperimentalUnsignedTypes::class)
fun UByteArray(data: ByteArray) = data.toUByteArray()
infix fun Int.hasFlag(v: Int): Boolean = (this and v) == v
