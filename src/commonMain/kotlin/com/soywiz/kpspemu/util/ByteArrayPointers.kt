package com.soywiz.kpspemu.util

import com.soywiz.kmem.*
import com.soywiz.korio.lang.*
import com.soywiz.korio.stream.*

abstract class p_void(val ba: ByteArray, val pos: Int) {
    override fun equals(other: Any?): Boolean {
        if (other !is p_void) return false
        return ba === other.ba && pos == other.pos
    }

    fun writeBytes(bytes: ByteArray, offset: Int = 0) {
        arraycopy(bytes, 0, ba, pos + offset, ba.size)
    }

    fun writeStringz(str: String, charset: Charset = UTF8) {
        writeBytes(str.toByteArray(charset) + byteArrayOf(0))
    }

    override fun hashCode(): Int = ba.hashCode() + pos
}

abstract class p_base<T>(ba: ByteArray, pos: Int, val esize: Int) : p_void(ba, pos) {
    abstract protected fun create(ba: ByteArray, pos: Int): T
    operator fun plus(offset: Int): T = create(ba, pos + offset)
}

class p_u8(ba: ByteArray, pos: Int) : p_base<p_u8>(ba, pos, 1) {
    override fun create(ba: ByteArray, pos: Int): p_u8 = p_u8(ba, pos)
    operator fun get(offset: Int) = ba[this.pos + offset].toInt() and 0xFF
    operator fun set(offset: Int, value: Int) = run { ba[this.pos + offset] = value.toByte() }
}

class p_u32(ba: ByteArray, pos: Int) : p_base<p_u32>(ba, pos, 4) {
    override fun create(ba: ByteArray, pos: Int): p_u32 = p_u32(ba, pos)
    operator fun get(offset: Int) = ba.readS32_le(pos + offset * 4)
    operator fun set(offset: Int, value: Int) = ba.write32_le(pos + offset * 4, value)
}

fun memcpy(dst: p_void, src: p_void, num: Int) = arraycopy(src.ba, src.pos, dst.ba, dst.pos, num)
fun memset(dst: p_void, value: Int, num: Int) = dst.ba.fill(value.toByte(), dst.pos, dst.pos + num)

fun memcmp(ptr1: p_void, ptr2: p_void, num: Int): Int {
    val p1 = ptr1.p_u8()
    val p2 = ptr2.p_u8()
    for (n in 0 until num) {
        val c1 = p1[n]
        val c2 = p2[n]
        if (c1 < c2) return -1
        if (c1 > c2) return +1
    }
    return 0
}

fun UByteArray.p_u8() = p_u8(this.data, 0)
fun ByteArray.p_u8() = p_u8(this, 0)

fun UByteArray.p_u32() = p_u32(this.data, 0)
fun ByteArray.p_u32() = p_u32(this, 0)

fun p_void.p_u8(): p_u8 = p_u8(ba, pos)
fun p_void.p_u32(): p_u32 = p_u32(ba, pos)
fun p_void.openSync(len: Int): SyncStream = this.ba.openSync().sliceWithSize(this.pos, len)

fun ByteArray.unsigned() = UByteArray(this)