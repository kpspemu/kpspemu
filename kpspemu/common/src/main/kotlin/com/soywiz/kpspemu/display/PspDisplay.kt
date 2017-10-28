package com.soywiz.kpspemu.display

import com.soywiz.kpspemu.mem.Memory

class PspDisplay(val mem: Memory) {
	var address: Int = 0x44000000
	var bufferWidth: Int = 512
	var pixelFormat: Int = 0
	var sync: Int = 0

	var displayMode: Int = 0
	var displayWidth: Int = 512
	var displayHeight: Int = 272
}