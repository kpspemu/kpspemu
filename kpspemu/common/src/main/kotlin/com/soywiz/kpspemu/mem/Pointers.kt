package com.soywiz.kpspemu.mem

import com.soywiz.korio.lang.format
import com.soywiz.korio.stream.SyncStream
import com.soywiz.korio.stream.SyncStreamBase
import com.soywiz.korio.stream.toSyncStream
import com.soywiz.kpspemu.util.unsigned

interface Ptr {
	val addr: Int
	fun sb(offset: Int, value: Int): Unit
	fun sh(offset: Int, value: Int): Unit
	fun sw(offset: Int, value: Int): Unit
	fun lb(offset: Int): Int
	fun lh(offset: Int): Int
	fun lw(offset: Int): Int

	fun sdw(offset: Int, value: Long): Unit {
		sw(offset + 0, (value ushr 0).toInt())
		sw(offset + 4, (value ushr 32).toInt())
	}

	fun ldw(offset: Int): Long {
		val low = lw(offset + 0).unsigned
		val high = lw(offset + 4).unsigned
		return (high shl 32) or low
	}
}

data class MemPtr(val mem: Memory, override val addr: Int) : Ptr {
	override fun sb(offset: Int, value: Int): Unit = mem.sb(addr + offset, value)
	override fun sh(offset: Int, value: Int): Unit = mem.sh(addr + offset, value)
	override fun sw(offset: Int, value: Int): Unit = mem.sw(addr + offset, value)
	override fun lb(offset: Int): Int = mem.lb(addr + offset)
	override fun lh(offset: Int): Int = mem.lh(addr + offset)
	override fun lw(offset: Int): Int = mem.lw(addr + offset)
	override fun toString(): String = "Ptr(0x%08X)".format(addr)
}

fun Memory.ptr(addr: Int) = MemPtr(this, addr)

val Ptr.isNotNull: Boolean get() = addr != 0
val Ptr.isNull: Boolean get() = addr == 0

fun Ptr.readBytes(count: Int, offset: Int = 0): ByteArray {
	val out = ByteArray(count)
	for (n in 0 until count) out[n] = this.lb(offset + n).toByte()
	return out
}

fun Ptr.openSync(): SyncStream {
	return object : SyncStreamBase() {
		override var length: Long = Long.MAX_VALUE
		override fun close() = Unit
		override fun read(position: Long, buffer: ByteArray, offset: Int, len: Int): Int {
			val start = position.toInt()
			for (n in 0 until len) buffer[offset + n] = lb(start + n).toByte()
			return len
		}

		override fun write(position: Long, buffer: ByteArray, offset: Int, len: Int) {
			val start = position.toInt()
			for (n in 0 until len) sb(start + n, buffer[offset + n].toInt())
		}
	}.toSyncStream(0L)
}