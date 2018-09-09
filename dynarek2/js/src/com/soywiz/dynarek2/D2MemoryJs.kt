package com.soywiz.dynarek2

import org.khronos.webgl.*

actual class D2Memory(val buffer: ArrayBuffer) {
    val s8 = Int8Array(buffer)
    val s16 = Int16Array(buffer)
    val s32 = Int32Array(buffer)
    val f32 = Float32Array(buffer)

    actual fun get8(index: Int): Int = s8[index].toInt()
    actual fun get16(index: Int): Int = s16[index].toInt()
    actual fun get32(index: Int): Int  = s32[index]
    actual fun getF32(index: Int): Float  = f32[index]

    actual fun set8(index: Int, value: Int) { s8[index] = value.toByte() }
    actual fun set16(index: Int, value: Int) { s16[index] = value.toShort() }
    actual fun set32(index: Int, value: Int) { s32[index] = value }
    actual fun setF32(index: Int, value: Float) { f32[index] = value }

    actual fun free() {
    }

}

actual fun NewD2Memory(size: Int): D2Memory = D2Memory(ArrayBuffer(size))
