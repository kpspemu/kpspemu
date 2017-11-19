package com.soywiz.kpspemu.display

import com.soywiz.korim.bitmap.Bitmap32
import com.soywiz.korim.color.RGBA
import com.soywiz.korim.color.RGBA_5551
import com.soywiz.korio.async.Signal
import com.soywiz.korio.async.waitOne
import com.soywiz.kpspemu.Emulator
import com.soywiz.kpspemu.WithEmulator
import com.soywiz.kpspemu.display
import com.soywiz.kpspemu.ge.PixelFormat
import com.soywiz.kpspemu.mem

class PspDisplay(override val emulator: Emulator) : WithEmulator {
	companion object {
		const val PROCESSED_PIXELS_PER_SECOND = 9000000 // hz
		const val CYCLES_PER_PIXEL = 1
		const val PIXELS_IN_A_ROW = 525
		const val VSYNC_ROW = 272
		const val NUMBER_OF_ROWS = 286
		const val HCOUNT_PER_VBLANK = 285.72
		const val HORIZONTAL_SYNC_HZ = (PspDisplay.PROCESSED_PIXELS_PER_SECOND * PspDisplay.CYCLES_PER_PIXEL) / PspDisplay.PIXELS_IN_A_ROW // 17142.85714285714
		const val HORIZONTAL_SECONDS = 1 / PspDisplay.HORIZONTAL_SYNC_HZ // 5.8333333333333E-5
		const val VERTICAL_SYNC_HZ = PspDisplay.HORIZONTAL_SYNC_HZ / PspDisplay.HCOUNT_PER_VBLANK // 59.998800024
		const val VERTICAL_SECONDS = 1 / PspDisplay.VERTICAL_SYNC_HZ // 0.016667
	}
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

	val onVsyncStart = Signal<Unit>()

	fun fixedAddress(): Int {
		//println(address.hex)
		return address
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

	fun crash() {
		for (n in 0 until 512 * 272) {
			mem.sw(address + n * 4, mem.lw(address + n * 4).inv())
		}
		//mem.fill(0, Memory.VIDEOMEM.start, Memory.VIDEOMEM.size)
	}

	suspend fun waitVblankStart() {
		display.onVsyncStart.waitOne()
	}

	suspend fun waitVblank() {
		//if (!inVBlank) {
			display.onVsyncStart.waitOne()
		//}
	}

	var inVBlank = false

	fun startVsync() {
		onVsyncStart(Unit)
		inVBlank = true
		vcount++
	}

	fun endVsync() {
		inVBlank = false
	}
}