package com.soywiz.kpspemu.hle.modules

import com.soywiz.kpspemu.mem.Ptr

class sceCtrl : SceModule() {
	override fun registerModule() {
		registerFunctionInt("sceCtrlPeekBufferPositive", 0x3A622550, 150, syscall = 0x2150) { sceCtrlPeekBufferPositive(ptr, int) }
	}

	fun sceCtrlPeekBufferPositive(sceCtrlDataPtr: Ptr, count: Int): Int {
		//console.log('sceCtrlPeekBufferPositive');
		for (n in 0 until count) {
			sceCtrlDataPtr.sw(n * 4, 0)
		}
		//return waitAsync(1).then(v => count);
		return count;
	}
}