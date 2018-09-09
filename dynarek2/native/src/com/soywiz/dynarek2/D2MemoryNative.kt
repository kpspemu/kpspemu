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

actual fun D2Memory.get8(index: Int): Int = s8[index].toInt()
actual fun D2Memory.get16(index: Int): Int = s16[index].toInt()
actual fun D2Memory.get32(index: Int): Int  = s32[index]
actual fun D2Memory.getF32(index: Int): Float  = f32[index]
actual fun D2Memory.set8(index: Int, value: Int) { s8[index] = value.toByte() }
actual fun D2Memory.set16(index: Int, value: Int) { s16[index] = value.toShort() }
actual fun D2Memory.set32(index: Int, value: Int) { s32[index] = value }
actual fun D2Memory.setF32(index: Int, value: Float) { f32[index] = value }

actual fun NewD2Memory(size: Int): D2MemoryFreeable {
    return object : D2MemoryFreeable {
        val ptr = mmap(
            null, size.convert(),
            PROT_READ or PROT_WRITE or PROT_EXEC,
            MAP_ANONYMOUS or MAP_SHARED, -1, 0
        )?.reinterpret<ByteVar>() ?: error("Couldn't reserve memory")

        override val mem: D2Memory = D2Memory(ptr)
        //override val mem: D2Memory = ptr.reinterpret<ByteVar>()

        override fun free() {
            munmap(ptr, size.convert())
        }
    }
}
