package com.soywiz.kpspemu

import com.soywiz.korio.mem.FastMemory

class Memory {
	companion object {
		const val MASK = 0x0FFFFFFF;

		val SCRATCHPAD = MemorySegment("scatchpad", 0x0000000 until 0x00010000)
		val VIDEOMEM = MemorySegment("videomem", 0x04000000 until 0x4200000)
		val MAINMEM = MemorySegment("mainmem", 0x08000000 until 0xa000000)
	}


	data class MemorySegment(val name: String, val range: IntRange) {
		val start get() = range.start
		val end get() = range.endInclusive + 1
	}

	private val buffer = FastMemory.alloc(0x0a000000)

	fun lb(address: Int) = buffer[(address and MASK)]
	fun lbu(address: Int) = buffer[(address and MASK)] and 0xFF

	fun lh(address: Int) = buffer.getAlignedInt16((address and MASK) ushr 1).toInt()
	fun lhu(address: Int) = buffer.getAlignedInt16((address and MASK) ushr 1).toInt() and 0xFFFF

	fun lw(address: Int) = buffer.getAlignedInt32((address and MASK) ushr 2)
}