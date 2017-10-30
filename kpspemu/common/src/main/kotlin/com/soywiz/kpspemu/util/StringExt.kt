package com.soywiz.kpspemu.util

import com.soywiz.korio.lang.format
import com.soywiz.korio.util.quote

fun String.parseInt(): Int = when {
	this.startsWith("0x", ignoreCase = true) -> this.substring(2).toLong(16).toInt()
	else -> this.toInt()
}

val String.quoted: String get() = this.quote()

val Int.hex: String get() = "0x%08X".format(this)