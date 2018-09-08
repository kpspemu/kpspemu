package com.soywiz.kmem.unsafe

import kotlinx.cinterop.*

val DataPointer.s8: CPointer<ByteVar> get() = buffer
val DataPointer.s16: CPointer<ShortVar> get() = buffer.reinterpret<ShortVar>()
val DataPointer.s32: CPointer<IntVar> get() = buffer.reinterpret<IntVar>()

actual class DataPointer(val buffer: CPointer<ByteVar>) {
    actual fun alignedGetS8(offset: Int): Byte = s8[offset]
    actual fun alignedGetS16(offset: Int): Short = s16[offset]
    actual fun alignedGetS32(offset: Int): Int = s32[offset]

    actual fun alignedSetS8(offset: Int, value: Byte) = run { s8[offset] = value }
    actual fun alignedSetS16(offset: Int, value: Short) = run { s16[offset] = value }
    actual fun alignedSetS32(offset: Int, value: Int) = run { s32[offset] = value }
    actual fun close() {}

}

actual fun NewDataPointer(size: Int): DataPointer {
    val data = ByteArray(size)
    val pin = data.pin()
    return DataPointer(pin.addressOf(0))
}
