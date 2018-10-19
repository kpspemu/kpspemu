package com.soywiz.kpspemu.mem

import com.soywiz.klogger.*
import com.soywiz.kmem.*
import com.soywiz.kmem.get
import com.soywiz.kmem.set
import com.soywiz.korio.crypto.*
import com.soywiz.korio.lang.*
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

abstract class Memory protected constructor(dummy: Boolean) {
    companion object {
        val MASK = MEMORY_MASK
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

        operator fun invoke(): Memory = com.soywiz.kpspemu.mem.NormalMemory()
        //operator fun invoke(): Memory = com.soywiz.kpspemu.mem.SmallMemory()
        //operator fun invoke(): Memory = SmallMemory()
    }

    data class MemorySegment(val name: String, val range: IntRange) {
        val start get() = range.start
        val end get() = range.endInclusive + 1
        val size get() = end - start
        operator fun contains(index: Int) = range.contains(index and MEMORY_MASK)
    }

    open fun hash(address4: Int, nwords: Int): Int {
        var hash = 0
        for (n in 0 until nwords) hash += lw(address4 + n * 4)
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

    open fun read(srcPos: Int, dst: ShortArray, dstPos: Int = 0, len: Int = dst.size - dstPos): Unit {
        for (n in 0 until len) dst[dstPos + n] = lh(srcPos + n * 4).toShort()
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

    open fun getFastMem(): KmlNativeBuffer? = null
    open fun getFastMemOffset(addr: Int): Int = 0

    abstract fun sb(address: Int, value: Int): Unit
    abstract fun sh(address: Int, value: Int): Unit
    abstract fun sw(address: Int, value: Int): Unit

    abstract fun lb(address: Int): Int
    abstract fun lh(address: Int): Int
    abstract fun lw(address: Int): Int

    inline fun svl_q(address: Int, read: (index: Int) -> Int) {
        val k = (3 - ((address ushr 2) and 3))
        var addr = address and 0xF.inv()
        for (n in k until 4) {
            sw(addr, read(n))
            addr += 4
        }
    }

    inline fun svr_q(address: Int, read: (index: Int) -> Int) {
        val k = (4 - ((address ushr 2) and 3))
        var addr = address
        for (n in 0 until k) {
            sw(addr, read(n))
            addr += 4
        }
    }

    inline fun lvl_q(address: Int, writer: (index: Int, value: Int) -> Unit) {
        val k = (3 - ((address ushr 2) and 3))
        var addr = address and 0xF.inv()
        for (n in k until 4) {
            writer(n, lw(addr))
            addr += 4
        }
    }

    inline fun lvr_q(address: Int, writer: (index: Int, value: Int) -> Unit) {
        val k = (4 - ((address ushr 2) and 3))
        var addr = address
        for (n in 0 until k) {
            writer(n, lw(addr))
            addr += 4
        }
    }

    // Unsigned
    fun lbu(address: Int): Int = lb(address) and 0xFF

    fun lhu(address: Int): Int = lh(address) and 0xFFFF
    fun memset(address: Int, value: Int, size: Int) = run { for (n in 0 until size) sb(address, value) }

    fun lwSafe(address: Int): Int = if (isValidAddress(address)) lw(address) else 0
    fun lbuSafe(address: Int): Int = if (isValidAddress(address)) lbu(address) else 0

    fun isValidAddress(address: Int): Boolean = address in MAINMEM || address in VIDEOMEM || address in SCRATCHPAD

    open fun copy(srcPos: Int, dstPos: Int, size: Int) = run { for (n in 0 until size) sb(dstPos + n, lb(srcPos + n)) }
    fun getPointerStream(address: Int, size: Int): SyncStream = openSync().sliceWithSize(address, size)
    fun getPointerStream(address: Int): SyncStream = openSync().sliceStart(address.toLong())
    fun readStringzOrNull(offset: Int): String? = if (offset != 0) readStringz(offset) else null

    fun readStringz(offset: Int): String = openSync().sliceStart(offset.toLong()).readStringz()
    open fun fill(value: Int, address: Int, size: Int) = run { for (n in 0 until size) sb(address + n, value) }

    fun reset() {
        for (seg in listOf(SCRATCHPAD, MAINMEM, VIDEOMEM)) {
            fill(0, seg.start, seg.size)
        }
    }
}

object DummyMemory : Memory(true) {
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

open class ListenerMemory(
    val parent: Memory = Memory(),
    val listener: (size: Int, addr: Int, old: Int, new: Int) -> Unit
) : Memory(true) {
    fun normalized(address: Int) = address and 0x0FFFFFFF

    private inline fun <T> check(address: Int, callback: () -> T): T {
        try {
            return callback()
        } catch (e: IndexOutOfBoundsException) {
            Console.error("OUT OF BOUNDS trying to read: ${address.hex}")
            throw e
        }
    }

    override fun sb(address: Int, value: Int) {
        check(address) {
            listener(1, address, parent.lb(address), value)
            parent.sb(normalized(address), value)
        }
    }

    override fun sh(address: Int, value: Int) {
        check(address) {
            listener(2, address, parent.lh(address), value)
            parent.sh(normalized(address), value)
        }
    }

    override fun sw(address: Int, value: Int) {
        check(address) {
            listener(4, address, parent.lw(address), value)
            parent.sw(normalized(address), value)
        }
    }

    override fun lb(address: Int): Int = check(address) { parent.lb(normalized(address)) }
    override fun lh(address: Int): Int = check(address) { parent.lh(normalized(address)) }
    override fun lw(address: Int): Int = check(address) { parent.lw(normalized(address)) }
}

abstract class FastMemoryBacked(val fmem: KmlNativeBuffer) : Memory(true) {
    protected abstract fun index(address: Int): Int

    val i8 = fmem.i8
    val i16 = fmem.i16
    val i32 = fmem.i32

    //val buffer = FastMemory.alloc(0x10000000)
    //private inline fun index(address: Int) = address and 0x0fffffff

    override fun hash(address4: Int, nwords: Int): Int {
        var hash = 0
        val i32 = this.i32
        val ptr = index(address4) ushr 2
        for (n in 0 until nwords) hash += i32[ptr + n]
        return hash
    }

    override fun sb(address: Int, value: Int) = run { i8[index(address)] = value.toByte() }
    override fun sh(address: Int, value: Int) = run { i16[index(address) ushr 1] = value.toShort() }
    override fun sw(address: Int, value: Int) = run { i32[index(address) ushr 2] = value }

    override fun lb(address: Int) = i8[index(address)].toInt()
    override fun lh(address: Int) = i16[index(address) ushr 1].toInt()
    override fun lw(address: Int): Int = i32[index(address) ushr 2]

    override fun copy(srcPos: Int, dstPos: Int, size: Int) =
        run { arraycopy(fmem.buffer, index(srcPos), fmem.buffer, index(dstPos), size) }

    override fun read(srcPos: Int, dst: ByteArray, dstPos: Int, len: Int): Unit =
        fmem.getArrayInt8(index(srcPos), dst, dstPos, len)

    override fun read(srcPos: Int, dst: ShortArray, dstPos: Int, len: Int): Unit =
        fmem.getAlignedArrayInt16(index(srcPos) ushr 1, dst, dstPos, len)

    override fun read(srcPos: Int, dst: IntArray, dstPos: Int, len: Int): Unit =
        fmem.getAlignedArrayInt32(index(srcPos) ushr 2, dst, dstPos, len)

    override fun write(dstPos: Int, src: ByteArray, srcPos: Int, len: Int) =
        fmem.setAlignedArrayInt8(index(dstPos), src, srcPos, len)

    override fun write(dstPos: Int, src: IntArray, srcPos: Int, len: Int) =
        fmem.setAlignedArrayInt32(index(dstPos) ushr 2, src, srcPos, len)

    override fun fill(value: Int, address: Int, size: Int) {
        val m = this.i8
        val start = index(address)
        val vb = value.toByte()
        // @TODO: Use native fill!
        for (n in start until start + size) m[n] = vb
    }

    override fun getFastMem(): KmlNativeBuffer? = fmem
    override fun getFastMemOffset(addr: Int): Int = index(addr)
}

class NormalMemory : FastMemoryBacked(KmlNativeBuffer.alloc(0x0a000000)) {
    override fun index(address: Int) = address and 0x0FFFFFFF
}

class SmallMemory :
    FastMemoryBacked(KmlNativeBuffer.alloc(Memory.MAINMEM.size + Memory.VIDEOMEM.size + Memory.SCRATCHPAD.size)) {
    override fun index(address: Int): Int {
        val addr = address and 0x0FFFFFFF
        return when {
            addr >= Memory.MAIN_OFFSET -> (addr - Memory.MAIN_OFFSET) // MAIN
            addr >= Memory.VIDEO_OFFSET -> (addr - Memory.VIDEO_OFFSET) + Memory.MAIN_SIZE
            else -> addr + Memory.MAIN_SIZE + +Memory.VIDEO_SIZE
        }
    }
}
