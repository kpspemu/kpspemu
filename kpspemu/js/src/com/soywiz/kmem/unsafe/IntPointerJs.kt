package com.soywiz.kmem.unsafe

import org.khronos.webgl.*

actual class IntPointer(val data: Int32Array) {
    actual operator fun get(index: Int): Int = data[index]
    actual operator fun set(index: Int, value: Int) = run { data[index] = value }
}

actual fun NewIntPointer(size: Int): IntPointer = IntPointer(Int32Array(size))