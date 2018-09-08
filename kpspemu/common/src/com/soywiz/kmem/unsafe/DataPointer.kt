package com.soywiz.kmem.unsafe

expect class DataPointer {
    fun alignedGetS8(offset: Int): Byte
    fun alignedGetS16(offset: Int): Short
    fun alignedGetS32(offset: Int): Int

    fun alignedSetS8(offset: Int, value: Byte)
    fun alignedSetS16(offset: Int, value: Short)
    fun alignedSetS32(offset: Int, value: Int)

    fun close()
}

expect fun NewDataPointer(size: Int): DataPointer
