package com.soywiz.kpspemu.util

import com.soywiz.korio.lang.format

val Int.shex: String get() = "%08X".format(this)
