package com.soywiz.kpspemu.mem

import com.soywiz.kmem.*
import com.soywiz.korio.error.*
import com.soywiz.korio.lang.*
import com.soywiz.korio.stream.*
import com.soywiz.kpspemu.util.*
import com.soywiz.krypto.encoding.*
import com.soywiz.korio.error.invalidOp as invalidOp1

data class PtrArray(val ptr: Ptr, val size: Int) {
    val addr: Int get() = ptr.addr
    val low: Int get() = ptr.addr
    val high: Int get() = low + size
    override fun toString(): String = "PtrArray($ptr, $size)"
}

interface BasePtr {
    val ptr: Ptr
}

class Ptr8(override val ptr: Ptr) : BasePtr {
    fun get(): Int = this[0]
    fun set(value: Int) = run { this[0] = value }
    operator fun get(index: Int): Int = ptr.lb(index)
    operator fun set(index: Int, value: Int) = ptr.sb(index, value)
    operator fun plus(offset: Int) = Ptr32(ptr + offset * 1)
    override fun toString(): String = "Ptr8($ptr)"
}

class Ptr32(override val ptr: Ptr) : BasePtr {
    fun get(): Int = this[0]
    fun set(value: Int) = run { this[0] = value }
    operator fun get(index: Int): Int = ptr.lw(index * 4)
    operator fun set(index: Int, value: Int) = ptr.sw(index * 4, value)
    operator fun plus(offset: Int) = Ptr32(ptr + offset * 4)
    override fun toString(): String = "Ptr32($ptr)"
}

class Ptr64(override val ptr: Ptr) : BasePtr {
    fun get(): Long = this[0]
    fun set(value: Long) = run { this[0] = value }
    operator fun get(index: Int): Long = ptr.ldw(index * 8)
    operator fun set(index: Int, value: Long) = ptr.sdw(index * 8, value)
    operator fun plus(offset: Int) = Ptr64(ptr + offset * 8)
    override fun toString(): String = "Ptr64($ptr)"
}

class PtrStruct<T>(val kind: StructType<T>, override val ptr: Ptr) : BasePtr {
    val kindSize = kind.size

    fun get(): T = this[0]
    fun set(value: T) = run { this[0] = value }
    operator fun get(index: Int): T = kind.read(ptr.openSync(index * kindSize))
    operator fun set(index: Int, value: T) = kind.write(ptr.openSync(index * kindSize), value)
    operator fun plus(offset: Int) = PtrStruct(kind, ptr + offset * kindSize)

    override fun toString(): String = "PtrStruct($kind, $ptr)"
}

interface Ptr : BasePtr {
    override val ptr: Ptr
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

    operator fun plus(offset: Int): Ptr
}

object DummyPtr : Ptr {
    override val ptr = this
    override val addr: Int = 0
    override fun sb(offset: Int, value: Int) = Unit
    override fun sh(offset: Int, value: Int) = Unit
    override fun sw(offset: Int, value: Int) = Unit
    override fun lb(offset: Int): Int = 0
    override fun lh(offset: Int): Int = 0
    override fun lw(offset: Int): Int = 0
    override fun plus(offset: Int): Ptr = DummyPtr
    override fun toString(): String = "DummyPtr(${addr.hex})"
}

val nullPtr = DummyPtr

fun <T> Ptr.read(struct: Struct<T>): T = openSync().read(struct)
fun <T> Ptr.write(struct: Struct<T>, value: T): Unit = openSync().write(struct, value)

inline fun <T, TR> Ptr.capture(struct: Struct<T>, callback: (T) -> TR): TR {
    val ptr = this
    val obj = ptr.openSync().read(struct)
    try {
        return callback(obj)
    } finally {
        ptr.openSync().write(struct, obj)
    }
}

data class MemPtr(val mem: Memory, override val addr: Int) : Ptr {
    override val ptr = this
    override fun sb(offset: Int, value: Int): Unit = mem.sb(addr + offset, value)
    override fun sh(offset: Int, value: Int): Unit = mem.sh(addr + offset, value)
    override fun sw(offset: Int, value: Int): Unit = mem.sw(addr + offset, value)
    override fun lb(offset: Int): Int = mem.lb(addr + offset)
    override fun lh(offset: Int): Int = mem.lh(addr + offset)
    override fun lw(offset: Int): Int = mem.lw(addr + offset)
    override fun toString(): String = "MemPtr(${addr.hex})"
    override fun plus(offset: Int): Ptr = MemPtr(mem, addr + offset)
}

fun Ptr.array(size: Int) = PtrArray(this, size)

fun Memory.ptr(addr: Int) = MemPtr(this, addr)

val BasePtr.isNotNull: Boolean get() = ptr.addr != 0
val BasePtr.isNull: Boolean get() = ptr.addr == 0

fun Ptr.writeBytes(bytes: ByteArray, offset: Int = 0, size: Int = bytes.size - offset) {
    for (n in 0 until size) this.sb(n, bytes[offset + n].toInt())
}

fun Ptr.readBytes(count: Int, offset: Int = 0): ByteArray {
    val out = ByteArray(count)
    for (n in 0 until count) out[n] = this.lb(offset + n).toByte()
    return out
}

fun Ptr.readStringz(charset: Charset = UTF8): String {
    val out = ByteArrayBuilder()
    var n = 0
    while (true) {
        val c = this.lb(n++)
        if (c == 0) break
        out.append(c.toByte())
        if (out.size >= 0x1000) invalidOp1("String is too big!")
    }
    return out.toByteArray().toString(charset)
}

fun Ptr.writeStringz(str: String, charset: Charset = UTF8): Unit {
    writeBytes(str.toByteArray(charset) + byteArrayOf(0))
}

fun Ptr.openSync(offset: Int = 0): SyncStream {
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
    }.toSyncStream(offset.toLong())
}
