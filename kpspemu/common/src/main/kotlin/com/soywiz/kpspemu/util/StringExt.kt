package com.soywiz.kpspemu.util

import com.soywiz.korio.lang.*

val Int.shex: String get() = "%08X".format(this)
