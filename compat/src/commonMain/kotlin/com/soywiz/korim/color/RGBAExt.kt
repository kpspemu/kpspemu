package com.soywiz.korim.color

fun RGBA.Companion.packFast(r: Int, g: Int, b: Int, a: Int): Int {
    return RGBA(r, g, b, a).value
}

val RGBA.rgba get() = value
