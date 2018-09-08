package com.soywiz.kmem.unsafe

import org.khronos.webgl.*

actual class DataPointer(val data: ArrayBuffer) {
    val s8 = Int8Array(data)
    val s16 = Int16Array(data)
    val s32 = Int32Array(data)

    actual fun alignedGetS8(offset: Int): Byte = s8[offset]
    actual fun alignedGetS16(offset: Int): Short = s16[offset]
    actual fun alignedGetS32(offset: Int): Int = s32[offset]

    actual fun alignedSetS8(offset: Int, value: Byte) { s8[offset] = value }
    actual fun alignedSetS16(offset: Int, value: Short) { s16[offset] = value }
    actual fun alignedSetS32(offset: Int, value: Int) { s32[offset] = value }
    actual fun close() {

    }
}

actual fun NewDataPointer(size: Int): DataPointer = DataPointer(ArrayBuffer(size))
