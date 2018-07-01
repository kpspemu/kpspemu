package com.soywiz.kpspemu.mem

import com.soywiz.korio.error.*
import com.soywiz.korio.stream.*

fun Memory.openSync(): SyncStream {
    val mem = this
    return SyncStream(object : SyncStreamBase() {
        override var length: Long get() = 0xFFFFFFFFL; set(value) = invalidOp

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
