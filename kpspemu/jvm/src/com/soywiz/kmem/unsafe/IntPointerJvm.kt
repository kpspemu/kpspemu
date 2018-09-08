package com.soywiz.kmem.unsafe

actual class IntPointer(val data: IntArray) {
    actual operator fun get(index: Int): Int = data[index]
    actual operator fun set(index: Int, value: Int) = run { data[index] = value }
}

actual fun NewIntPointer(size: Int): IntPointer = IntPointer(IntArray(size))