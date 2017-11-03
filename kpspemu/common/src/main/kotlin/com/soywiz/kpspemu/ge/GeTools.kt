package com.soywiz.kpspemu.ge

import com.soywiz.korim.bitmap.Bitmap32
import com.soywiz.korim.color.PaletteColorFormat

fun PixelFormat.decode(out: Bitmap32, colorData: ByteArray): Bitmap32 {
	val cf = this.colorFormat ?: TODO("Temporally unsupported decoding of palletized or DXT")
	cf.decodeToBitmap32(out, colorData)
	return out
}

