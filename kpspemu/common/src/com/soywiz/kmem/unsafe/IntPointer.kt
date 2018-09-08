package com.soywiz.kmem.unsafe

expect class IntPointer {
    operator fun get(index: Int): Int
    operator fun set(index: Int, value: Int)
}

expect fun NewIntPointer(size: Int): IntPointer

fun IntPointer.getFloat(index: Int): Float = Float.fromBits(this.get(index))
fun IntPointer.setFloat(index: Int, value: Float) = this.set(index, value.toRawBits())
