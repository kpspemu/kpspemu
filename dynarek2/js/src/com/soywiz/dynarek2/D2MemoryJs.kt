package com.soywiz.dynarek2

import org.khronos.webgl.*

actual class D2Memory(val buffer: ArrayBuffer) : D2MemoryFreeable {
    override val mem = this

    @JsName("s8")
    val s8 = Int8Array(buffer)

    @JsName("s16")
    val s16 = Int16Array(buffer)

    @JsName("s32")
    val s32 = Int32Array(buffer)

    @JsName("s64")
    val s64 = Int64ArrayImpl(buffer)

    @JsName("f32")
    val f32 = Float32Array(buffer)

    override fun free() {
    }
}

class Int64ArrayImpl(val buffer: ArrayBuffer) {
    @JsName("s32")
    val s32 = Int32Array(buffer)

    operator fun get(index: Int): Long {
        val low = s32[index * 2 + 0]
        val high = s32[index * 2 + 1]
        return (high.toLong() shl 32) or low.toLong()
    }

    operator fun set(index: Int, value: Long) {
        s32[index * 2 + 0] = (value shr 0).toInt()
        s32[index * 2 + 1] = (value shr 32).toInt()
    }
}

actual fun D2Memory.get8(index: Int): Int = s8[index].toInt()
actual fun D2Memory.get16(index: Int): Int = s16[index].toInt()
actual fun D2Memory.get32(index: Int): Int  = s32[index]
actual fun D2Memory.getF32(index: Int): Float  = f32[index]
actual fun D2Memory.set8(index: Int, value: Int) { s8[index] = value.toByte() }
actual fun D2Memory.set16(index: Int, value: Int) { s16[index] = value.toShort() }
actual fun D2Memory.set32(index: Int, value: Int) { s32[index] = value }
actual fun D2Memory.setF32(index: Int, value: Float) { f32[index] = value }

actual fun NewD2Memory(size: Int): D2MemoryFreeable = D2Memory(ArrayBuffer(size))
