package com.soywiz.kpspemu.format

import com.soywiz.kds.*
import com.soywiz.kmem.*
import com.soywiz.korio.compression.*
import com.soywiz.korio.compression.deflate.*
import com.soywiz.korio.error.*
import com.soywiz.korio.file.*
import com.soywiz.korio.stream.*
import kotlin.math.*

class Cso private constructor() {
    val CACHED_BLOCKS = 0x10

    lateinit var s: AsyncStream
    var magic: String = ""
    var headerSize: Int = 0
    var totalBytes: Long = 0L
    var blockSize: Int = 0
    var version: Int = 0
    var alignment: Int = 0
    var reserved: Int = 0
    var offsets = IntArray(0)
    var numberOfBlocks: Int = 0

    suspend private fun init(s: AsyncStream) {
        this.s = s
        magic = s.readStringz(4)
        if (magic != "CISO") invalidOp("Not a CISO file")
        headerSize = s.readS32_le()
        totalBytes = s.readS64_le()
        blockSize = s.readS32_le()
        version = s.readU8()
        alignment = s.readU8()
        reserved = s.readS16_le()
        //println(s.position)
        //println(headerSize)
        numberOfBlocks = ceil(totalBytes.toDouble() / blockSize.toDouble()).toInt()
        offsets = s.readIntArray_le(numberOfBlocks + 1)
    }

    private fun isValidBlock(block: Int): Boolean = block in 0 until (offsets.size - 1)
    private fun isBlockUncompressed(block: Int): Boolean = (offsets[block] hasFlag 0x80000000L.toInt())

    suspend fun readCompressedBlock(block: Int): ByteArray {
        val start = offsets[block] and 0x7FFFFFFF
        val end = offsets[block + 1] and 0x7FFFFFFF
        return s.sliceWithBounds(start.toLong(), end.toLong()).readAll()
    }

    suspend fun readUncompressedBlock(block: Int): ByteArray {
        //println("readUncompressedBlock")
        if (isBlockUncompressed(block)) {
            return readCompressedBlock(block)
        } else {
            return readCompressedBlock(block).syncUncompress(Deflate)
        }
    }

    val blockCache = CacheMap<Int, ByteArray>(CACHED_BLOCKS)

    suspend fun readUncompressedBlockCached(block: Int): ByteArray {
        return blockCache.getOrPut(block) {
            readUncompressedBlock(block)
        }
    }

    companion object {
        suspend operator fun invoke(s: AsyncStream) = Cso().apply { init(s) }
    }

    //val compressedStream: AsyncStream get() = s
    suspend fun open(): AsyncStream {
        return object : AsyncStreamBase() {
            override suspend fun close() = s.close()
            override suspend fun setLength(value: Long) = invalidOp("Unsupported")
            override suspend fun getLength(): Long = totalBytes

            suspend fun readChunk(position: Long, buffer: ByteArray, offset: Int, len: Int): Int {
                //if (len <= 0) return 0
                //println("readChunk: $position, $buffer, $offset, $len")
                val block = (position / blockSize).toInt()
                val pinblock = (position % blockSize).toInt()
                if (isValidBlock(block)) {
                    val blockData = readUncompressedBlockCached(block)
                    val toRead = min(len, blockData.size - pinblock)
                    arraycopy(blockData, pinblock, buffer, offset, toRead)
                    return toRead
                } else {
                    return 0
                }
            }

            suspend fun readComplete(position: Long, buffer: ByteArray, offset: Int, len: Int): Int {
                val available = getLength() - position
                var cposition = position
                var pending = min(available, len.toLong()).toInt()
                var coffset = offset
                var tread = 0
                while (pending >= 0) {
                    val read = readChunk(cposition, buffer, coffset, pending)
                    if (read <= 0) break
                    pending -= read
                    cposition += read
                    coffset += read
                    tread += read
                }
                return tread
            }

            suspend override fun read(position: Long, buffer: ByteArray, offset: Int, len: Int): Int {
                //return readChunk(position, buffer, offset, len)
                return readComplete(position, buffer, offset, len)
            }

            suspend override fun write(position: Long, buffer: ByteArray, offset: Int, len: Int) =
                invalidOp("Unsupported")
        }.toAsyncStream()
    }
}

suspend fun AsyncStream.openAsCso() = Cso(this).open()
suspend fun VfsFile.openAsCso() = Cso(this.open()).open()
