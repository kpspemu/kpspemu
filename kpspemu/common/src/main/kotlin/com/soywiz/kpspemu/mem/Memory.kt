package com.soywiz.kpspemu.mem

import com.soywiz.korio.error.invalidOp
import com.soywiz.korio.lang.Console
import com.soywiz.korio.lang.format
import com.soywiz.korio.mem.FastMemory
import com.soywiz.korio.stream.*
import com.soywiz.kpspemu.util.hex

const private val MEMORY_MASK = 0x0FFFFFFF
const private val MASK = MEMORY_MASK

private val LWR_MASK = intArrayOf(0x00000000, 0xFF000000.toInt(), 0xFFFF0000.toInt(), 0xFFFFFF00.toInt())
private val LWR_SHIFT = intArrayOf(0, 8, 16, 24)

private val LWL_MASK = intArrayOf(0x00FFFFFF, 0x0000FFFF, 0x000000FF, 0x00000000)
private val LWL_SHIFT = intArrayOf(24, 16, 8, 0)

private val SWL_MASK = intArrayOf(0xFFFFFF00.toInt(), 0xFFFF0000.toInt(), 0xFF000000.toInt(), 0x00000000)
private val SWL_SHIFT = intArrayOf(24, 16, 8, 0)

private val SWR_MASK = intArrayOf(0x00000000, 0x000000FF, 0x0000FFFF, 0x00FFFFFF)
private val SWR_SHIFT = intArrayOf(0, 8, 16, 24)

abstract class Memory protected constructor(dummy: Boolean) {
	companion object {
		val MASK = MEMORY_MASK
		const val MAIN_OFFSET = 0x08000000
		const val HIGH_MASK = 0xf0000000.toInt()

		val SCRATCHPAD = MemorySegment("scatchpad", 0x0000000 until 0x00010000)
		val VIDEOMEM = MemorySegment("videomem", 0x04000000 until 0x4200000)
		val MAINMEM = MemorySegment("mainmem", MAIN_OFFSET until 0x0a000000)

		operator fun invoke(): Memory = com.soywiz.kpspemu.mem.FastMemory()
		//operator fun invoke(): Memory = SmallMemory()
	}

	data class MemorySegment(val name: String, val range: IntRange) {
		val start get() = range.start
		val end get() = range.endInclusive + 1
		val size get() = end - start
	}

	open fun hash(address4: Int, nwords: Int): Int {
		var hash = 0
		for (n in 0 until nwords) {
			hash += lw(address4 + n * 4)
		}
		return hash
	}

	fun readBytes(srcPos: Int, count: Int): ByteArray = ByteArray(count).apply { read(srcPos, this, 0, count) }

	open fun write(dstPos: Int, src: ByteArray, srcPos: Int = 0, len: Int = src.size - srcPos): Unit {
		for (n in 0 until len) sb(dstPos + n, src[srcPos + n].toInt())
	}

	open fun read(srcPos: Int, dst: ByteArray, dstPos: Int = 0, len: Int = dst.size - dstPos): Unit {
		for (n in 0 until len) dst[dstPos + n] = this.lb(srcPos + n).toByte()
	}

	open fun write(dstPos: Int, src: IntArray, srcPos: Int = 0, len: Int = src.size - srcPos): Unit {
		for (n in 0 until len) sw(dstPos + n * 4, src[srcPos + n].toInt())
	}

	open fun read(srcPos: Int, dst: IntArray, dstPos: Int = 0, len: Int = dst.size - dstPos): Unit {
		for (n in 0 until len) dst[dstPos + n] = lw(srcPos + n * 4)
	}

	fun lwl(address: Int, value: Int): Int {
		val align = address and 3
		val oldvalue = this.lw(address and 3.inv())
		return ((oldvalue shl LWL_SHIFT[align]) or (value and LWL_MASK[align]))
	}

	fun lwr(address: Int, value: Int): Int {
		val align = address and 3
		val oldvalue = this.lw(address and 3.inv())
		return ((oldvalue ushr LWR_SHIFT[align]) or (value and LWR_MASK[align]))
	}

	fun swl(address: Int, value: Int): Unit {
		val align = address and 3
		val aadress = address and 3.inv()
		val vwrite = (value ushr SWL_SHIFT[align]) or (this.lw(aadress) and SWL_MASK[align])
		this.sw(aadress, vwrite)
	}

	fun swr(address: Int, value: Int): Unit {
		val align = address and 3
		val aadress = address and 3.inv()
		val vwrite = (value shl SWR_SHIFT[align]) or (this.lw(aadress) and SWR_MASK[align])
		this.sw(aadress, vwrite)
	}

	abstract fun sb(address: Int, value: Int): Unit
	abstract fun sh(address: Int, value: Int): Unit
	abstract fun sw(address: Int, value: Int): Unit

	abstract fun lb(address: Int): Int
	abstract fun lh(address: Int): Int
	abstract fun lw(address: Int): Int

	// Unsigned
	fun lbu(address: Int): Int = lb(address) and 0xFF

	fun lhu(address: Int): Int = lh(address) and 0xFFFF
	fun memset(address: Int, value: Int, size: Int) {
		for (n in 0 until size) sb(address, value)
	}

	open fun copy(srcPos: Int, dstPos: Int, size: Int) = run { for (n in 0 until size) sb(dstPos + n, lb(srcPos + n)) }

	fun getPointerStream(address: Int, size: Int): SyncStream {
		return openSync().sliceWithSize(address, size)
	}

	fun readStringzOrNull(offset: Int): String? = when {
		offset == 0 -> null
		else -> readStringz(offset)
	}

	fun readStringz(offset: Int): String = openSync().sliceWithStart(offset.toLong()).readStringz()
	fun fill(value: Int, offset: Int, size: Int) {
		for (n in 0 until offset) sb(offset + n, value)
	}
}

class DummyMemory : Memory(true) {
	override fun sb(address: Int, value: Int) = Unit
	override fun sh(address: Int, value: Int) = Unit
	override fun sw(address: Int, value: Int) = Unit
	override fun lb(address: Int): Int = 0
	override fun lh(address: Int): Int = 0
	override fun lw(address: Int): Int = 0
}

fun Memory.trace(traceWrites: Boolean = true, traceReads: Boolean = false) = TraceMemory(this, traceWrites, traceReads)

class TraceMemory(
	val parent: Memory = Memory(),
	val traceWrites: Boolean = true,
	val traceReads: Boolean = false
) : Memory(true) {
	fun normalized(address: Int) = address and MASK

	override fun sb(address: Int, value: Int) {
		if (traceWrites) println("sb(0x%08X) = %d".format(normalized(address), value))
		parent.sb(address, value)
	}

	override fun sh(address: Int, value: Int) {
		if (traceWrites) println("sh(0x%08X) = %d".format(normalized(address), value))
		parent.sh(address, value)
	}

	override fun sw(address: Int, value: Int) {
		if (traceWrites) println("sw(0x%08X) = %d".format(normalized(address), value))
		parent.sw(address, value)
	}

	override fun lb(address: Int): Int {
		if (traceReads) println("lb(0x%08X)".format(normalized(address)))
		val res = parent.lb(address)
		if (traceReads) println("-> %d".format(res))
		return res
	}

	override fun lh(address: Int): Int {
		if (traceReads) println("lh(0x%08X)".format(normalized(address)))
		val res = parent.lh(address)
		if (traceReads) println("-> %d".format(res))
		return res
	}

	override fun lw(address: Int): Int {
		if (traceReads) println("lw(0x%08X)".format(normalized(address)))
		val res = parent.lw(address)
		if (traceReads) println("-> %d".format(res))
		return res
	}
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

abstract class FastMemoryBacked : Memory(true) {
	protected abstract val buffer: com.soywiz.korio.mem.FastMemory
	protected abstract fun index(address: Int): Int

	//val buffer = FastMemory.alloc(0x10000000)
	//private inline fun index(address: Int) = address and 0x0fffffff

	override fun hash(address4: Int, nwords: Int): Int {
		var hash = 0
		val buffer = this.buffer
		val ptr = index(address4) ushr 2
		for (n in 0 until nwords) hash += buffer.getAlignedInt32(ptr + n)
		return hash
	}

	override fun sb(address: Int, value: Int) = run { buffer[index(address)] = value }
	override fun sh(address: Int, value: Int) = run { buffer.setAlignedInt16(index(address) ushr 1, value.toShort()) }
	override fun sw(address: Int, value: Int) = run { buffer.setAlignedInt32(index(address) ushr 2, value) }

	override fun lb(address: Int) = buffer[index(address)]
	override fun lh(address: Int) = buffer.getAlignedInt16(index(address) ushr 1).toInt()
	override fun lw(address: Int): Int = buffer.getAlignedInt32(index(address) ushr 2)

	override fun copy(srcPos: Int, dstPos: Int, size: Int) = FastMemory.copy(buffer, index(srcPos), buffer, index(dstPos), size)
	override fun read(srcPos: Int, dst: ByteArray, dstPos: Int, len: Int): Unit = buffer.getArrayInt8(index(srcPos), dst, dstPos, len)
	override fun read(srcPos: Int, dst: IntArray, dstPos: Int, len: Int): Unit = buffer.getAlignedArrayInt32(index(srcPos) ushr 2, dst, dstPos, len)
	override fun write(dstPos: Int, src: ByteArray, srcPos: Int, len: Int) = buffer.setAlignedArrayInt8(index(dstPos), src, srcPos, len)
	override fun write(dstPos: Int, src: IntArray, srcPos: Int, len: Int) = buffer.setAlignedArrayInt32(index(dstPos) ushr 2, src, srcPos, len)
}

class FastMemory : FastMemoryBacked() {
	override val buffer = com.soywiz.korio.mem.FastMemory.alloc(0x0a000000)
	override fun index(address: Int) = address and MASK
}

class SmallMemory : FastMemoryBacked() {
	override val buffer = FastMemory.alloc(0x02000000 + 0x0200000 + 0x00010000)

	override fun index(address: Int): Int = when {
		address >= 0x08000000 -> address - 0x08000000
		address >= 0x04000000 -> address - 0x04000000 + 0x02000000
		else -> address + 0x04000000 + 0x02000000
	}
}
