package com.soywiz.dynarek2

import java.nio.*

actual class D2Memory(val buffer: ByteBuffer) {
    val s8 = buffer
    val s16 = buffer.asShortBuffer()
    val s32 = buffer.asIntBuffer()
    val f32 = buffer.asFloatBuffer()

    actual fun get8(index: Int): Int = s8[index].toInt()
    actual fun get16(index: Int): Int = s16[index].toInt()
    actual fun get32(index: Int): Int  = s32[index]
    actual fun getF32(index: Int): Float  = f32[index]

    actual fun set8(index: Int, value: Int) { s8.put(index, value.toByte()) }
    actual fun set16(index: Int, value: Int) { s16.put(index, value.toShort()) }
    actual fun set32(index: Int, value: Int) { s32.put(index, value) }
    actual fun setF32(index: Int, value: Float) { f32.put(index, value) }

    actual fun free() {
    }

}

actual fun NewD2Memory(size: Int): D2Memory = D2Memory(ByteBuffer.allocateDirect(size).order(ByteOrder.LITTLE_ENDIAN))
