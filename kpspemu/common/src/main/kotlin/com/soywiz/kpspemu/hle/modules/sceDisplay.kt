package com.soywiz.kpspemu.hle.modules

import com.soywiz.kpspemu.Emulator
import com.soywiz.kpspemu.hle.PixelFormat
import com.soywiz.kpspemu.hle.SceModule
import com.soywiz.kpspemu.hle.manager.PspThread
import com.soywiz.kpspemu.hle.manager.WaitObject

class sceDisplay(emulator: Emulator) : SceModule(emulator, "sceDisplay") {
	val display by lazy { emulator.display }

	override fun registerModule() {
		registerFunctionInt("sceDisplaySetMode", 0x0E20F177, 150, syscall = 0x213A) { sceDisplaySetMode(int, int, int) }
		registerFunctionInt("sceDisplayWaitVblankStart", 0x984C27E7, 150, syscall = 0x2147) { sceDisplayWaitVblankStart(thread) }
		registerFunctionInt("sceDisplaySetFrameBuf", 0x289D82FE, 150, syscall = 0x213F) { sceDisplaySetFrameBuf(int, int, int, int) }
	}

	fun sceDisplaySetMode(mode: Int, width: Int, height: Int): Int {
		//console.info(sprintf("sceDisplay.sceDisplaySetMode(mode: %d, width: %d, height: %d)", mode, width, height));
		display.displayMode = mode;
		display.displayWidth = width
		display.displayHeight = height
		return 0
	}

	fun sceDisplayWaitVblankStart(thread: PspThread): Int {
		//println("sceDisplayWaitVblankStart")
		thread.suspend(WaitObject.VBLANK, cb = false)
		//return this._waitVblankAsync(thread, AcceptCallbacks.NO);
		//return this._waitVblankStartAsync(thread, AcceptCallbacks.NO);
		return 0
	}

	fun sceDisplaySetFrameBuf(address: Int, bufferWidth: Int, pixelFormat: Int, sync: Int): Int {
		// PixelFormat
		//println("display.address: $address")
		display.address = address
		display.bufferWidth = bufferWidth
		display.pixelFormat = PixelFormat(pixelFormat)
		display.sync = sync
		return 0
	}
}