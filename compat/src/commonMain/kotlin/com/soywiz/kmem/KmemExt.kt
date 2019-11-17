package com.soywiz.kmem

import com.soywiz.korio.util.encoding.*

typealias KmlNativeBuffer = MemBuffer

val MemBuffer.i32 get() = this.asInt32Buffer()
