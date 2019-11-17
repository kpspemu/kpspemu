package com.soywiz.korio.compression

fun ByteArray.syncUncompress(algo: CompressionMethod): ByteArray {
    return this.uncompress(algo)
}