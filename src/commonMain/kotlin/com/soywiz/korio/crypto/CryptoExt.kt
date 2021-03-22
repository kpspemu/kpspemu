package com.soywiz.korio.crypto

import com.soywiz.korio.util.encoding.*

typealias Hex = com.soywiz.korio.util.encoding.Hex

val ByteArray.hex get() = Hex.encodeLower(this)
val Int.hex: String get() = "0x$shex"
val Int.shex: String
    get() {
        var out = ""
        for (n in 0 until 8) {
            val v = (this ushr ((7 - n) * 4)) and 0xF
            out += Hex.encodeCharUpper(v)
        }
        return out
    }
