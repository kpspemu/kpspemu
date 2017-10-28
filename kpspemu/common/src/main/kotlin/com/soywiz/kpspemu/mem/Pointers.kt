package com.soywiz.kpspemu.mem

import com.soywiz.korio.lang.format

interface Ptr {
	val addr: Int
	fun sb(offset: Int, value: Int): Unit
	fun sh(offset: Int, value: Int): Unit
	fun sw(offset: Int, value: Int): Unit
	fun lb(offset: Int): Int
	fun lh(offset: Int): Int
	fun lw(offset: Int): Int
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