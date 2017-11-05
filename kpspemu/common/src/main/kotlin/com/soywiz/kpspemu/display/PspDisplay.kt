package com.soywiz.kpspemu.display

import com.soywiz.korim.bitmap.Bitmap32
import com.soywiz.korim.color.RGBA
import com.soywiz.korim.color.RGBA_5551
import com.soywiz.korio.async.Signal
import com.soywiz.kpspemu.Emulator
import com.soywiz.kpspemu.WithEmulator
import com.soywiz.kpspemu.ge.PixelFormat
import com.soywiz.kpspemu.mem
import com.soywiz.kpspemu.util.hex

class PspDisplay(override val emulator: Emulator) : WithEmulator {
	var exposeDisplay = true

	val bmp = Bitmap32(512, 272)

	var rawDisplay: Boolean = true

	var address: Int = 0x44000000
	var bufferWidth: Int = 512
	var pixelFormat: PixelFormat = PixelFormat.RGBA_8888
	var sync: Int = 0

	var displayMode: Int = 0
	//var displayWidth: Int = 512
	var displayWidth: Int = 480
	var displayHeight: Int = 272

	var vcount = 0

	val onVsync = Signal<Unit>()

	fun fixedAddress(): Int {
		//println(address.hex)
		return address
	}

	fun dispatchVsync() {
		vcount++
		onVsync(Unit)
	}

	private val temp = ByteArray(512 * 272 * 4)

	fun decodeToBitmap32(out: Bitmap32) {
		val bmpData = out.data

		when (pixelFormat) {
			PixelFormat.RGBA_8888 -> { // Optimized!
				mem.read(address, bmpData)
			}
			else -> {
				mem.read(address, temp, 0, temp.size)
				val color = when (pixelFormat) {
				//PixelFormat.RGBA_5650 -> RGBA
					PixelFormat.RGBA_5551 -> RGBA_5551
					else -> RGBA
				}

				color.decodeToBitmap32(out, temp)
				//RGBA_4444.decodeToBitmap32(out, temp)
				//RGBA.decodeToBitmap32(out, temp)
			}
		}
	}

	fun clear() {
		for (n in 0 until 512 * 272) {
			mem.sw(address + n * 4, mem.lw(address + n * 4).inv())
		}
		//mem.fill(0, Memory.VIDEOMEM.start, Memory.VIDEOMEM.size)
	}
}