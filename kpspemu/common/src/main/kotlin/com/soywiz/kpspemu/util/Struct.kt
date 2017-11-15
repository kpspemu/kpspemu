package com.soywiz.kpspemu.util

import com.soywiz.korio.lang.Charset
import com.soywiz.korio.stream.*
import kotlin.reflect.KMutableProperty1

open class Struct<T>(val create: () -> T, vararg val items: Item<T, *>) : StructType<T> {
	data class Item<T1, V>(val type: StructType<V>, val property: KMutableProperty1<T1, V>)

	companion object {
		infix fun <T1, V> KMutableProperty1<T1, V>.AS(type: StructType<V>) = Item(type, this)
	}

	override fun write(s: SyncStream, value: T) {
		for (item in items) {
			@Suppress("UNCHECKED_CAST")
			val i = item as Item<T, Any>
			i.type.write(s, i.property.get(value))
		}
	}

	override fun read(s: SyncStream): T = read(s, create())

	fun read(s: SyncStream, value: T): T {
		for (item in items) {
			@Suppress("UNCHECKED_CAST")
			val i = item as Item<T, Any>
			i.property.set(value, i.type.read(s))
		}
		return value
	}
}

/*
class Header(
	var magic: Int = 0,
	var size: Int = 0
) {
	companion object : Struct<Header>({ Header() },
		Item(INT32, Header::magic),
		Item(INT32, Header::size)
	)
}

fun test() {
	Header.write()
}
*/

interface StructType<T> {
	fun write(s: SyncStream, value: T)
	fun read(s: SyncStream): T
}

fun <T> SyncStream.write(s: StructType<T>, value: T) = s.write(this, value)
fun <T> SyncStream.read(s: StructType<T>): T = s.read(this)

object UINT8 : StructType<Int> {
	override fun write(s: SyncStream, value: Int) = s.write8(value)
	override fun read(s: SyncStream): Int = s.readU8()
}

object INT32 : StructType<Int> {
	override fun write(s: SyncStream, value: Int) = s.write32_le(value)
	override fun read(s: SyncStream): Int = s.readS32_le()
}

object FLOAT32 : StructType<Float> {
	override fun write(s: SyncStream, value: Float) = s.writeF32_le(value)
	override fun read(s: SyncStream): Float = s.readF32_le()
}

class STRINGZ(val charset: Charset, val size: Int? = null) : StructType<String> {
	override fun write(s: SyncStream, value: String) = when {
		size == null -> s.writeStringz(value, charset)
		else -> s.writeStringz(value, size, charset)
	}

	override fun read(s: SyncStream): String = when {
		size == null -> s.readStringz(charset)
		else -> s.readStringz(size, charset)
	}
}
