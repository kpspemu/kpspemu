package com.soywiz.kpspemu.hle.modules

import com.soywiz.kpspemu.PspThread

class sceDisplay : SceModule() {
	val display by lazy { e.display }

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
		//return this._waitVblankAsync(thread, AcceptCallbacks.NO);
		//return this._waitVblankStartAsync(thread, AcceptCallbacks.NO);
		return 0
	}

	fun sceDisplaySetFrameBuf(address: Int, bufferWidth: Int, pixelFormat: Int, sync: Int): Int {
		// PixelFormat
		//println("display.address: $address")
		display.address = address
		display.bufferWidth = bufferWidth
		display.pixelFormat = pixelFormat
		display.sync = sync
		return 0
	}
}