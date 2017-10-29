package com.soywiz.kpspemu.display

import com.soywiz.korim.bitmap.Bitmap32
import com.soywiz.korim.color.RGBA
import com.soywiz.korim.color.RGBA_5551
import com.soywiz.kpspemu.hle.PixelFormat
import com.soywiz.kpspemu.mem.Memory

class PspDisplay(val mem: Memory) {
	var rawDisplay: Boolean = true

	var address: Int = 0x44000000
	var bufferWidth: Int = 512
	var pixelFormat: PixelFormat = PixelFormat.RGBA_8888
	var sync: Int = 0

	var displayMode: Int = 0
	//var displayWidth: Int = 512
	var displayWidth: Int = 480
	var displayHeight: Int = 272

	fun dispatchVsync() {
	}

	private val temp = ByteArray(512 * 272 * 4)

	fun decodeToBitmap32(out: Bitmap32) {
		//println(pixelFormat)
		mem.read(address, temp, 0, temp.size)

		val color = when (pixelFormat) {
			//PixelFormat.RGBA_5650 -> RGBA
			PixelFormat.RGBA_5551 -> RGBA_5551
			else -> RGBA
		}

		color.decodeToBitmap32(out, temp)
		//RGBA_4444.decodeToBitmap32(out, temp)
		//RGBA.decodeToBitmap32(out, temp)
		val bmpData = out.data
		for (n in 0 until bmpData.size) bmpData[n] = (bmpData[n] and 0x00FFFFFF) or 0xFF000000.toInt()
	}
}