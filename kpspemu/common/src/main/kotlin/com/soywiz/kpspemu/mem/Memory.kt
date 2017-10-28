package com.soywiz.kpspemu.mem

import com.soywiz.korio.error.invalidOp
import com.soywiz.korio.mem.FastMemory
import com.soywiz.korio.stream.SyncStream
import com.soywiz.korio.stream.SyncStreamBase

const private val MASK = 0x0FFFFFFF;

interface Memory {
	companion object {
		val SCRATCHPAD = MemorySegment("scatchpad", 0x0000000 until 0x00010000)
		val VIDEOMEM = MemorySegment("videomem", 0x04000000 until 0x4200000)
		val MAINMEM = MemorySegment("mainmem", 0x08000000 until 0x0a000000)

		//operator fun invoke(): Memory = com.soywiz.kpspemu.mem.FastMemory()
		operator fun invoke(): Memory = SmallMemory()
	}

	data class MemorySegment(val name: String, val range: IntRange) {
		val start get() = range.start
		val end get() = range.endInclusive + 1
	}

	fun read(srcPos: Int, dst: ByteArray, dstPos: Int = 0, len: Int = dst.size - dstPos): Unit {
		for (n in 0 until len) dst[dstPos + n] = this.lb(srcPos + n).toByte()
	}

	fun readBytes(srcPos: Int, count: Int): ByteArray = ByteArray(count).apply { read(srcPos, this, 0, count) }

	fun write(dstPos: Int, src: ByteArray, srcPos: Int = 0, len: Int = src.size - srcPos): Unit {
		for (n in 0 until len) sb(dstPos + n, src[srcPos + n].toInt())
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

class DummyMemory : Memory {
	override fun sb(address: Int, value: Int) = Unit
	override fun sh(address: Int, value: Int) = Unit
	override fun sw(address: Int, value: Int) = Unit
	override fun lb(address: Int): Int = 0
	override fun lbu(address: Int): Int = 0
	override fun lh(address: Int): Int = 0
	override fun lhu(address: Int): Int = 0
	override fun lw(address: Int): Int = 0
}

fun Memory.openSync(): SyncStream {
	val mem = this
	return SyncStream(object : SyncStreamBase() {
		override var length: Long
			get() = 0xFFFFFFFFL
			set(value) = invalidOp

		override fun close() = Unit
		override fun read(position: Long, buffer: ByteArray, offset: Int, len: Int): Int {
			mem.read(position.toInt(), buffer, offset, len)
			return len
		}

		override fun write(position: Long, buffer: ByteArray, offset: Int, len: Int) {
			mem.write(position.toInt(), buffer, offset, len)
		}
	})
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
