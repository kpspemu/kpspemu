package com.soywiz.kpspemu.ge

import com.soywiz.korim.bitmap.Bitmap32

fun Bitmap32.setTo(colorFormat: PixelFormat, colorData: ByteArray, clutData: ByteArray? = null, clutFormat: PixelFormat? = null, clutColors: Int = 0): Bitmap32 {
	when {
		colorFormat.requireClut -> {
			//println(clutFormat)
			val clutBmp = Bitmap32(clutColors, 1)
			clutBmp.setTo(clutFormat!!, clutData!!)
			val colors = clutBmp.data

			when (colorFormat.paletteBits) {
				4 -> {
					//println("PAL4")
					var m = 0
					for (n in 0 until this.area / 2) {
						val byte = colorData[n].toInt() and 0xFF
						this.data[m++] = colors[(byte ushr 0) and 0b1111]
						this.data[m++] = colors[(byte ushr 4) and 0b1111]
					}
				}
				8 -> {
					var m = 0
					for (n in 0 until this.area) {
						val byte = colorData[n]
						this.data[m++] = colors[(byte.toInt() ushr 0) and 0b11111111]
					}
				}
			}
		}
		colorFormat.isRgba -> {
			val cf = colorFormat.colorFormat!!
			cf.decodeToBitmap32(this, colorData)
		}
		colorFormat.isCompressed -> {
			TODO("Unsupported DXT")
		}
	}
	return this
}

