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