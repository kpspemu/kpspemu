package com.soywiz.kpspemu

import com.soywiz.korio.mem.FastMemory

const private val MASK = 0x0FFFFFFF;

interface Memory {
	companion object {
		val SCRATCHPAD = MemorySegment("scatchpad", 0x0000000 until 0x00010000)
		val VIDEOMEM = MemorySegment("videomem", 0x04000000 until 0x4200000)
		val MAINMEM = MemorySegment("mainmem", 0x08000000 until 0x0a000000)

		//operator fun invoke(): Memory = com.soywiz.kpspemu.FastMemory()
		operator fun invoke(): Memory = com.soywiz.kpspemu.SmallMemory()
	}

	data class MemorySegment(val name: String, val range: IntRange) {
		val start get() = range.start
		val end get() = range.endInclusive + 1
	}

	fun sb(address: Int, value: Int): Unit
	fun sh(address: Int, value: Int): Unit
	fun sw(address: Int, value: Int): Unit

	fun lb(address: Int): Int
	fun lbu(address: Int): Int
	fun lh(address: Int): Int
	fun lhu(address: Int): Int
	fun lw(address: Int): Int
}

class FastMemory : Memory {
	private val buffer = FastMemory.alloc(0x0a000000)

	fun index(address: Int) = address and MASK

	override fun sb(address: Int, value: Int) = run { buffer[index(address)] = value }
	override fun sh(address: Int, value: Int) = run { buffer.setAlignedInt16(index(address), value.toShort()) }
	override fun sw(address: Int, value: Int) = run { buffer.setAlignedInt32(index(address), value) }

	override fun lb(address: Int) = buffer[index(address)]
	override fun lbu(address: Int) = buffer[index(address)] and 0xFF
	override fun lh(address: Int) = buffer.getAlignedInt16(index(address) ushr 1).toInt()
	override fun lhu(address: Int) = buffer.getAlignedInt16(index(address) ushr 1).toInt() and 0xFFFF
	override fun lw(address: Int) = buffer.getAlignedInt32(index(address) ushr 2)
}

class SmallMemory : Memory {
	private val buffer = FastMemory.alloc(0x02000000 + 0x0200000 + 0x00010000)

	fun index(address: Int): Int = when {
		address >= 0x08000000 -> address - 0x08000000
		address >= 0x04000000 -> address - 0x04000000 + 0x02000000
		else -> address + 0x04000000 + 0x02000000
	}

	override fun sb(address: Int, value: Int) = run { buffer[index(address)] = value }
	override fun sh(address: Int, value: Int) = run { buffer.setAlignedInt16(index(address), value.toShort()) }
	override fun sw(address: Int, value: Int) = run { buffer.setAlignedInt32(index(address), value) }

	override fun lb(address: Int) = buffer[index(address)]
	override fun lbu(address: Int) = buffer[index(address)] and 0xFF
	override fun lh(address: Int) = buffer.getAlignedInt16(index(address) ushr 1).toInt()
	override fun lhu(address: Int) = buffer.getAlignedInt16(index(address) ushr 1).toInt() and 0xFFFF
	override fun lw(address: Int) = buffer.getAlignedInt32(index(address) ushr 2)
}
