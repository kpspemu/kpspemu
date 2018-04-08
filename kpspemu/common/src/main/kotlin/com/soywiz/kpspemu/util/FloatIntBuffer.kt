package com.soywiz.kpspemu.util

import com.soywiz.kmem.*

class FloatIntBuffer(val size: Int) {
    val mem = MemBufferAlloc(size * 4)
    var f = mem.asFloat32Buffer()
    var i = mem.asInt32Buffer()
}
