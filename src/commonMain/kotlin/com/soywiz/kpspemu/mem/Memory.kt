package com.soywiz.kpspemu.mem

import com.soywiz.dynarek2.*
import com.soywiz.kmem.*
import com.soywiz.korio.stream.*

private const val MEMORY_MASK = 0x0FFFFFFF
private const val MASK = MEMORY_MASK

private val LWR_MASK = intArrayOf(0x00000000, 0xFF000000.toInt(), 0xFFFF0000.toInt(), 0xFFFFFF00.toInt())
private val LWR_SHIFT = intArrayOf(0, 8, 16, 24)

private val LWL_MASK = intArrayOf(0x00FFFFFF, 0x0000FFFF, 0x000000FF, 0x00000000)
private val LWL_SHIFT = intArrayOf(24, 16, 8, 0)

private val SWL_MASK = intArrayOf(0xFFFFFF00.toInt(), 0xFFFF0000.toInt(), 0xFF000000.toInt(), 0x00000000)
private val SWL_SHIFT = intArrayOf(24, 16, 8, 0)

private val SWR_MASK = intArrayOf(0x00000000, 0x000000FF, 0x0000FFFF, 0x00FFFFFF)
private val SWR_SHIFT = intArrayOf(0, 8, 16, 24)

typealias Memory = D2Memory

object MemoryInfo {
    const val MASK = MEMORY_MASK
    const val HIGH_MASK = 0xf0000000.toInt()
    inline val SCATCHPAD_OFFSET get() = 0x0000000
    inline val VIDEO_OFFSET get() = 0x04000000
    inline val MAIN_OFFSET get() = 0x08000000

    inline val SCATCHPAD_SIZE get() = 64 * 1024 // 64 KB
    inline val MAIN_SIZE get() = 32 * 1024 * 1024 // 32 MB
    inline val VIDEO_SIZE get() = 2 * 1024 * 1024 // 2 MB

    val SCRATCHPAD = MemorySegment("scatchpad", SCATCHPAD_OFFSET until (SCATCHPAD_OFFSET + SCATCHPAD_SIZE))
    val VIDEOMEM = MemorySegment("videomem", VIDEO_OFFSET until (VIDEO_OFFSET + VIDEO_SIZE))
    val MAINMEM = MemorySegment("mainmem", MAIN_OFFSET until (MAIN_OFFSET + MAIN_SIZE))

    val DUMMY: Memory by lazy { NewD2Memory(1024).mem }
}

data class MemorySegment(val name: String, val range: IntRange) {
    val start get() = range.start
    val end get() = range.endInclusive + 1
    val size get() = end - start
    operator fun contains(index: Int) = range.contains(index and MEMORY_MASK)
}

val cachedMemory by lazy { NewD2Memory(0x10000000).mem }
fun CachedMemory(): Memory = cachedMemory
fun Memory(): Memory = NewD2Memory(0x10000000).mem

fun Memory.readBytes(srcPos: Int, count: Int): ByteArray = ByteArray(count).apply { read(srcPos, this, 0, count) }

fun Memory.write(dstPos: Int, src: ByteArray, srcPos: Int = 0, len: Int = src.size - srcPos): Unit {
    for (n in 0 until len) sb(dstPos + n, src[srcPos + n].toInt())
}

fun Memory.read(srcPos: Int, dst: ByteArray, dstPos: Int = 0, len: Int = dst.size - dstPos): Unit {
    for (n in 0 until len) dst[dstPos + n] = this.lb(srcPos + n).toByte()
}

fun Memory.write(dstPos: Int, src: IntArray, srcPos: Int = 0, len: Int = src.size - srcPos): Unit {
    for (n in 0 until len) sw(dstPos + n * 4, src[srcPos + n].toInt())
}

fun Memory.read(srcPos: Int, dst: ShortArray, dstPos: Int = 0, len: Int = dst.size - dstPos): Unit {
    for (n in 0 until len) dst[dstPos + n] = lh(srcPos + n * 4).toShort()
}

fun Memory.read(srcPos: Int, dst: IntArray, dstPos: Int = 0, len: Int = dst.size - dstPos): Unit {
    for (n in 0 until len) dst[dstPos + n] = lw(srcPos + n * 4)
}

fun Memory.lwl(address: Int, value: Int): Int {
    val align = address and 3
    val oldvalue = this.lw(address and 3.inv())
    return ((oldvalue shl LWL_SHIFT[align]) or (value and LWL_MASK[align]))
}

fun Memory.lwr(address: Int, value: Int): Int {
    val align = address and 3
    val oldvalue = this.lw(address and 3.inv())
    return ((oldvalue ushr LWR_SHIFT[align]) or (value and LWR_MASK[align]))
}

fun Memory.swl(address: Int, value: Int): Unit {
    val align = address and 3
    val aadress = address and 3.inv()
    val vwrite = (value ushr SWL_SHIFT[align]) or (this.lw(aadress) and SWL_MASK[align])
    this.sw(aadress, vwrite)
}

fun Memory.swr(address: Int, value: Int): Unit {
    val align = address and 3
    val aadress = address and 3.inv()
    val vwrite = (value shl SWR_SHIFT[align]) or (this.lw(aadress) and SWR_MASK[align])
    this.sw(aadress, vwrite)
}

fun Memory.index(address: Int) = address and 0x0FFFFFFF

fun Memory.sb(address: Int, value: Int) = run { set8(index(address) ushr 0, value) }
fun Memory.sh(address: Int, value: Int) = run { set16(index(address) ushr 1, value) }
fun Memory.sw(address: Int, value: Int) = run { set32(index(address) ushr 2, value) }

fun Memory.lb(address: Int) = get8(index(address) ushr 0).toInt()
fun Memory.lh(address: Int) = get16(index(address) ushr 1).toInt()
fun Memory.lw(address: Int): Int = get32(index(address) ushr 2).toInt()

fun Memory.getFastMem(): KmlNativeBuffer? = null
fun Memory.getFastMemOffset(addr: Int): Int = index(addr)

fun Memory.svl_q(address: Int, read: (index: Int) -> Int) {
    val k = (3 - ((address ushr 2) and 3))
    var addr = address and 0xF.inv()
    for (n in k until 4) {
        sw(addr, read(n))
        addr += 4
    }
}

fun Memory.svr_q(address: Int, read: (index: Int) -> Int) {
    val k = (4 - ((address ushr 2) and 3))
    var addr = address
    for (n in 0 until k) {
        sw(addr, read(n))
        addr += 4
    }
}

fun Memory.lvl_q(address: Int, writer: (index: Int, value: Int) -> Unit) {
    val k = (3 - ((address ushr 2) and 3))
    var addr = address and 0xF.inv()
    for (n in k until 4) {
        writer(n, lw(addr))
        addr += 4
    }
}

fun Memory.lvr_q(address: Int, writer: (index: Int, value: Int) -> Unit) {
    val k = (4 - ((address ushr 2) and 3))
    var addr = address
    for (n in 0 until k) {
        writer(n, lw(addr))
        addr += 4
    }
}

// Unsigned
fun Memory.lbu(address: Int): Int = lb(address) and 0xFF

fun Memory.lhu(address: Int): Int = lh(address) and 0xFFFF

fun Memory.lwSafe(address: Int): Int = if (isValidAddress(address)) lw(address) else 0
fun Memory.lbuSafe(address: Int): Int = if (isValidAddress(address)) lbu(address) else 0

fun Memory.isValidAddress(address: Int): Boolean =
    address in MemoryInfo.MAINMEM || address in MemoryInfo.VIDEOMEM || address in MemoryInfo.SCRATCHPAD

fun Memory.getPointerStream(address: Int, size: Int): SyncStream = openSync().sliceWithSize(address, size)
fun Memory.getPointerStream(address: Int): SyncStream = openSync().sliceStart(address.toLong())
fun Memory.readStringzOrNull(offset: Int): String? = if (offset != 0) readStringz(offset) else null

//fun Memory.readStringz(offset: Int): String = openSync().sliceStart(offset.toLong()).readStringz()
fun Memory.strlenz(offset: Int): Int {
    val idx = this.index(offset)
    for (n in 0 until Int.MAX_VALUE) {
        val c = get8(idx + n)
        if (c == 0) return n
    }
    return -1
}

// @TODO: UTF-8
fun Memory.readStringz(offset: Int): String {
    val len = strlenz(offset)
    val idx = this.index(offset)
    return CharArray(len) { get8(idx + it).toChar() }.concatToString()
}

fun Memory.copy(srcPos: Int, dstPos: Int, size: Int) = run {
    val srcOffset = index(srcPos)
    val dstOffset = index(dstPos)
    for (n in 0 until size) {
        set8(dstOffset + n, get8(srcOffset + n))
    }
}

fun Memory.memset(address: Int, value: Int, size: Int) = fill(value, address, size)

fun Memory.fill(value: Int, address: Int, size: Int) {
    val offset = index(address)
    for (n in 0 until size) {
        set8(offset + n, value)
    }
}

fun Memory.hash(address4: Int, nwords: Int): Int {
    var hash = 0
    val offset4 = index(address4 * 4) / 4
    for (n in 0 until nwords) {
        hash += get32(offset4 + n)
    }
    return hash
}

fun Memory.reset() {
    for (seg in listOf(MemoryInfo.SCRATCHPAD, MemoryInfo.MAINMEM, MemoryInfo.VIDEOMEM)) {
        fill(0, seg.start, seg.size)
    }
}

//fun Memory.close() = close()
