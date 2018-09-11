package com.soywiz.dynarek2

import kotlinx.cinterop.*
import platform.posix.*

actual class D2Memory(val buffer: CPointer<ByteVar>)
//actual typealias D2Memory = CPointer<ByteVar>
//inline val D2Memory.buffer get() = this.reinterpret<ByteVar>()

inline val D2Memory.ptr get() = buffer.reinterpret<ByteVar>()
inline val D2Memory.s8 get() = buffer.reinterpret<ByteVar>()
inline val D2Memory.s16 get() = buffer.reinterpret<ShortVar>()
inline val D2Memory.s32 get() = buffer.reinterpret<IntVar>()
inline val D2Memory.f32 get() = buffer.reinterpret<FloatVar>()

actual inline fun D2Memory.get8(index: Int): Int = s8[index].toInt()
actual inline fun D2Memory.get16(index: Int): Int = s16[index].toInt()
actual inline fun D2Memory.get32(index: Int): Int  = s32[index]
actual inline fun D2Memory.getF32(index: Int): Float  = f32[index]
actual inline fun D2Memory.set8(index: Int, value: Int) { s8[index] = value.toByte() }
actual inline fun D2Memory.set16(index: Int, value: Int) { s16[index] = value.toShort() }
actual inline fun D2Memory.set32(index: Int, value: Int) { s32[index] = value }
actual inline fun D2Memory.setF32(index: Int, value: Float) { f32[index] = value }


