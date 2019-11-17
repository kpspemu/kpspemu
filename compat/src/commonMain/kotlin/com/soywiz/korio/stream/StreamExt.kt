package com.soywiz.korio.stream

import com.soywiz.kmem.*

fun SyncStream.readS16_le() = readS16LE()
fun SyncStream.readS16_be() = readS16BE()
fun SyncStream.readU16_le() = readU16LE()
fun SyncStream.readU16_be() = readU16BE()

fun SyncStream.readS32_le() = readS32LE()
fun SyncStream.readS32_be() = readS32BE()

fun SyncStream.readS64_le() = readS64LE()
fun SyncStream.readS64_be() = readS64BE()

fun SyncStream.readF32_le() = readF32LE()
fun SyncStream.readF32_be() = readF32BE()

fun SyncStream.write16_le(v: Int) = write16LE(v)
fun SyncStream.write16_be(v: Int) = write16BE(v)

fun SyncStream.write32_le(v: Int) = write32LE(v)
fun SyncStream.write32_be(v: Int) = write32BE(v)

fun SyncStream.write64_le(v: Long) = write64LE(v)
fun SyncStream.write64_be(v: Long) = write64BE(v)

fun SyncStream.writeF32_le(v: Float) = writeF32LE(v)
fun SyncStream.writeF32_be(v: Float) = writeF32BE(v)

fun SyncStream.readShortArray_le(len: Int): ShortArray = readShortArrayLE(len)
fun SyncStream.readShortArray_be(len: Int): ShortArray = readShortArrayBE(len)

fun SyncStream.readCharArray_le(len: Int): CharArray = readCharArrayLE(len)
fun SyncStream.readCharArray_be(len: Int): CharArray = readCharArrayBE(len)

fun SyncStream.readIntArray_le(len: Int): IntArray = readIntArrayLE(len)
fun SyncStream.readIntArray_be(len: Int): IntArray = readIntArrayBE(len)

fun SyncStream.writeShortArray_le(v: ShortArray) = writeShortArrayLE(v)
fun SyncStream.writeShortArray_be(v: ShortArray) = writeShortArrayBE(v)

fun SyncStream.writeCharArray_le(v: CharArray) = writeCharArrayLE(v)
fun SyncStream.writeCharArray_be(v: CharArray) = writeCharArrayBE(v)

fun SyncStream.writeIntArray_le(v: IntArray) = writeIntArrayLE(v)
fun SyncStream.writeIntArray_be(v: IntArray) = writeIntArrayBE(v)

fun ByteArray.readS32_le(o: Int) = this.readS32LE(o)
fun ByteArray.readS32_be(o: Int) = this.readS32BE(o)

fun ByteArray.write32_le(o: Int, v: Int) = this.write32LE(o, v)

suspend fun AsyncStream.readS16_le() = readS16LE()
suspend fun AsyncStream.readS32_le() = readS32LE()
suspend fun AsyncStream.readS64_le() = readS64LE()
suspend fun AsyncStream.readIntArray_le(count: Int) = readIntArrayLE(count)
