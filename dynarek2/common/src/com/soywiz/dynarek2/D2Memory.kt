package com.soywiz.dynarek2

expect class D2Memory {
	fun get8(index: Int): Int
	fun get16(index: Int): Int
	fun get32(index: Int): Int
	fun getF32(index: Int): Float

	fun set8(index: Int, value: Int): Unit
	fun set16(index: Int, value: Int): Unit
	fun set32(index: Int, value: Int): Unit
	fun setF32(index: Int, value: Float): Unit

	fun free()
}

expect fun NewD2Memory(size: Int): D2Memory

fun NewD2Memory(data: ByteArray): D2Memory {
	return NewD2Memory(data.size).apply {
		for (n in 0 until data.size) this.set8(n, data[n].toInt())
	}
}
