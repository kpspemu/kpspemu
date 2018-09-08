package com.soywiz.kmem.unsafe

import kotlinx.cinterop.*

actual class IntPointer(val i32: CPointer<IntVar>) {
    actual operator fun get(index: Int): Int = i32[index]
    actual operator fun set(index: Int, value: Int) = run { i32[index] = value }
}

actual fun NewIntPointer(size: Int): IntPointer {
    val data = IntArray(size)
    val pin = data.pin()
    return IntPointer(pin.addressOf(0))
}