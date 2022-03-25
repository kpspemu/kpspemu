package com.soywiz.kpspemu.util

import com.soywiz.korio.lang.*
import com.soywiz.korio.stream.*
import com.soywiz.korio.util.*
import kotlin.reflect.*

open class Struct<T>(val create: () -> T, vararg val items: Item<T, *>) : StructType<T> {
    override val size: Int = items.map { it.type.size }.sum()

    data class Item<T1, V>(val type: StructType<V>, val property: KMutableProperty1<T1, V>)

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

    fun toByteArray(value: T): ByteArray = MemorySyncStream().apply { write(this, value) }.toByteArray()
}

infix fun <T1, V> KMutableProperty1<T1, V>.AS(type: StructType<V>) = Struct.Item(type, this)

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
    val size: Int
    fun write(s: SyncStream, value: T)
    fun read(s: SyncStream): T
}

fun <T> SyncStream.write(s: StructType<T>, value: T) = s.write(this, value)
fun <T> SyncStream.read(s: StructType<T>): T = s.read(this)

object UINT8 : StructType<Int> {
    override val size = 1
    override fun write(s: SyncStream, value: Int) = s.write8(value)
    override fun read(s: SyncStream): Int = s.readU8()
}

object UINT16 : StructType<Int> {
    override val size = 2
    override fun write(s: SyncStream, value: Int) = s.write16_le(value)
    override fun read(s: SyncStream): Int = s.readU16_le()
}

object INT32 : StructType<Int> {
    override val size = 4
    override fun write(s: SyncStream, value: Int) = s.write32_le(value)
    override fun read(s: SyncStream): Int = s.readS32_le()
}

object INT32_be : StructType<Int> {
    override val size = 4
    override fun write(s: SyncStream, value: Int) = s.write32_be(value)
    override fun read(s: SyncStream): Int = s.readS32_be()
}


fun <T, TR> StructType<T>.map(map: (T) -> TR, invMap: (TR) -> T): StructType<TR> = object : StructType<TR> {
    override val size: Int = this@map.size
    override fun write(s: SyncStream, value: TR) = this@map.write(s, invMap(value))
    override fun read(s: SyncStream): TR = map(this@map.read(s))
}

fun <TR : IdEnum> StructType<Int>.asEnum(e: BaseEnum<TR>): StructType<TR> = this.map({ e(it) }, { it.id })

object INT64 : StructType<Long> {
    override val size = 8
    override fun write(s: SyncStream, value: Long) = s.write64_le(value)
    override fun read(s: SyncStream): Long = s.readS64_le()
}

object FLOAT32 : StructType<Float> {
    override val size = 4
    override fun write(s: SyncStream, value: Float) = s.writeF32_le(value)
    override fun read(s: SyncStream): Float = s.readF32_le()
}

class STRINGZ(val charset: Charset, val len: Int? = null) : StructType<String> {
    override val size = len ?: 0

    constructor(size: Int? = null) : this(UTF8, size)

    override fun write(s: SyncStream, value: String) = when {
        len == null -> s.writeStringz(value, charset)
        else -> s.writeStringz(value, len, charset)
    }

    override fun read(s: SyncStream): String = when {
        len == null -> s.readStringz(charset)
        else -> s.readStringz(len, charset)
    }
}

class ARRAY<T>(val etype: StructType<T>, val len: Int) : StructType<ArrayList<T>> {
    override val size: Int = len * etype.size

    override fun write(s: SyncStream, value: ArrayList<T>) {
        for (v in value) s.write(etype, v)
    }

    override fun read(s: SyncStream): ArrayList<T> {
        val out = arrayListOf<T>()
        for (n in 0 until len) out += s.read(etype)
        return out
    }
}

class INTLIKEARRAY(val etype: StructType<Int>, val len: Int) : StructType<IntArray> {
    override val size: Int = len * etype.size

    override fun write(s: SyncStream, value: IntArray) {
        for (v in value) s.write(etype, v)
    }

    override fun read(s: SyncStream): IntArray {
        val out = IntArray(len)
        for (n in 0 until len) out[n] = s.read(etype)
        return out
    }
}

class BYTEARRAY(val len: Int) : StructType<ByteArray> {
    override val size: Int = len
    override fun write(s: SyncStream, value: ByteArray) = s.writeBytes(value)
    override fun read(s: SyncStream): ByteArray = s.readBytes(len)
}

class SHORTARRAY(val len: Int) : StructType<ShortArray> {
    override val size: Int = 2 * len
    override fun write(s: SyncStream, value: ShortArray) = s.writeShortArray_le(value)
    override fun read(s: SyncStream): ShortArray = s.readShortArray_le(len)
}

class CHARARRAY(val len: Int) : StructType<CharArray> {
    override val size: Int = 2 * len
    override fun write(s: SyncStream, value: CharArray) = s.writeCharArray_le(value)
    override fun read(s: SyncStream): CharArray = s.readCharArray_le(len)
}

class INTARRAY(val len: Int) : StructType<IntArray> {
    override val size: Int = 4 * len
    override fun write(s: SyncStream, value: IntArray) = s.writeIntArray_le(value)
    override fun read(s: SyncStream): IntArray = s.readIntArray_le(len)
}

interface BaseEnum<T : IdEnum> {
    operator fun invoke(id: Int): T
}

open class INT32_ENUM<T : IdEnum>(val values: Array<T>) : StructType<T>, BaseEnum<T> {
    override val size: Int = 4

    override fun write(s: SyncStream, value: T) = s.write32_le(value.id)
    override fun read(s: SyncStream): T = invoke(s.readS32_le())

    private val defaultValue: T = values.first()
    private val MAX_ID = values.map { it.id }.maxOrNull() ?: 0
    private val valuesById = Array<Any>(MAX_ID + 1) { defaultValue }

    init {
        for (v in values) valuesById[v.id] = v
    }

    operator override fun invoke(id: Int): T = valuesById.getOrElse(id) { defaultValue } as T
}

open class UINT8_ENUM<T : IdEnum>(val values: Array<T>) : StructType<T>, BaseEnum<T> {
    override val size: Int = 1

    override fun write(s: SyncStream, value: T) = s.write8(value.id)
    override fun read(s: SyncStream): T = invoke(s.readU8())

    private val defaultValue: T = values.first()
    private val MAX_ID = values.map { it.id }.maxOrNull() ?: 0
    private val valuesById = Array<Any>(MAX_ID + 1) { defaultValue }

    init {
        for (v in values) valuesById[v.id] = v
    }

    operator override fun invoke(id: Int): T = valuesById.getOrElse(id) { defaultValue } as T
}