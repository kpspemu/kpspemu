package com.soywiz.dynarek2

expect class D2Memory

expect fun D2Memory.get8(index: Int): Int
expect fun D2Memory.get16(index: Int): Int
expect fun D2Memory.get32(index: Int): Int
expect fun D2Memory.getF32(index: Int): Float

expect fun D2Memory.set8(index: Int, value: Int): Unit
expect fun D2Memory.set16(index: Int, value: Int): Unit
expect fun D2Memory.set32(index: Int, value: Int): Unit
expect fun D2Memory.setF32(index: Int, value: Float): Unit

interface D2MemoryFreeable {
	val mem: D2Memory
	fun free()
}

expect fun NewD2Memory(size: Int): D2MemoryFreeable

fun NewD2Memory(data: ByteArray): D2MemoryFreeable {
	return NewD2Memory(data.size).apply {
		val mem = this.mem
		for (n in 0 until data.size) mem.set8(n, data[n].toInt())
	}
}

inline fun NewD2Memory(size: Int, callback: (D2Memory) -> Unit) {
	val mem = NewD2Memory(size)
	try {
		callback(mem.mem)
	} finally {
		mem.free()
	}
}

inline class D2MemoryInt(val mem: D2Memory)
inline operator fun D2MemoryInt.get(index: Int) = mem.get32(index)
inline operator fun D2MemoryInt.set(index: Int, value: Int) = mem.set32(index, value)

inline fun D2MemoryInt.getFloat(index: Int) = mem.getF32(index)
inline fun D2MemoryInt.setFloat(index: Int, value: Float) = mem.setF32(index, value)
