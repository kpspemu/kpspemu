package com.soywiz.korio.stream

fun SyncStream.readS16_le() = readS16LE()
fun SyncStream.readS16_be() = readS16BE()
fun SyncStream.readU16_le() = readU16LE()
fun SyncStream.readU16_be() = readU16BE()

fun SyncStream.readS32_le() = readS32LE()
fun SyncStream.readS32_be() = readS32BE()

fun SyncStream.readS64_le() = readS64LE()
fun SyncStream.readS64_be() = readS64BE()

fun SyncStream.write16_le(v: Int) = write16LE(v)
fun SyncStream.write32_le(v: Int) = write32LE(v)
fun SyncStream.write16_be(v: Int) = write16BE(v)
fun SyncStream.write32_be(v: Int) = write32BE(v)
