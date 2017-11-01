package com.soywiz.kpspemu.util

import com.soywiz.korim.bitmap.Bitmap32

fun Bitmap32.setAlpha(value: Int) {
	for (n in 0 until this.data.size) this.data[n] = (this.data[n] and 0x00FFFFFF) or (value shl 24)
}