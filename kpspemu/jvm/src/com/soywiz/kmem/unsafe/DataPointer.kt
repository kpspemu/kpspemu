package com.soywiz.kmem.unsafe

import java.nio.*

actual class DataPointer(val buffer: ByteBuffer) {
    val s8 = buffer
    val s16 = buffer.asShortBuffer()
    val s32 = buffer.asIntBuffer()

    actual fun alignedGetS8(offset: Int): Byte = s8[offset]
    actual fun alignedGetS16(offset: Int): Short = s16[offset]
    actual fun alignedGetS32(offset: Int): Int = s32[offset]

    actual fun alignedSetS8(offset: Int, value: Byte) { s8.put(offset, value) }
    actual fun alignedSetS16(offset: Int, value: Short) { s16.put(offset, value) }
    actual fun alignedSetS32(offset: Int, value: Int) { s32.put(offset, value) }
    actual fun close() {}

}

actual fun NewDataPointer(size: Int): DataPointer = DataPointer(ByteBuffer.allocateDirect(size).order(ByteOrder.nativeOrder()))
