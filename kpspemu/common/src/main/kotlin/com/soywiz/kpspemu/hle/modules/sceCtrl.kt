package com.soywiz.kpspemu.hle.modules

import com.soywiz.kpspemu.Emulator
import com.soywiz.kpspemu.controller
import com.soywiz.kpspemu.cpu.CpuState
import com.soywiz.kpspemu.hle.SceModule
import com.soywiz.kpspemu.mem.Ptr

class sceCtrl(emulator: Emulator) : SceModule(emulator, "sceCtrl", 0x40010011, "ctrl.prx", "sceController_Service") {
	fun sceCtrlPeekBufferPositive(sceCtrlDataPtr: Ptr, count: Int): Int {
		//console.log('sceCtrlPeekBufferPositive');
		var pos = 0
		for (n in 0 until count) {
			sceCtrlDataPtr.sw(pos + 0, 0) // timestamp
			sceCtrlDataPtr.sw(pos + 4, controller.buttons) // buttons // @TODO: forced button!
			sceCtrlDataPtr.sb(pos + 8, controller.lx) // lx
			sceCtrlDataPtr.sb(pos + 9, controller.ly) // ly
			pos += 16
		}
		//return waitAsync(1).then(v => count);
		return count;
	}

	fun sceCtrlReadBufferPositive(sceCtrlDataPtr: Ptr, count: Int): Int {
		return sceCtrlPeekBufferPositive(sceCtrlDataPtr, count)
	}

	fun sceCtrlSetSamplingCycle(samplingCycle: Int): Int {
		controller.samplingCycle = samplingCycle
		return 0
	}

	fun sceCtrlSetSamplingMode(samplingMode: Int): Int {
		controller.samplingMode = samplingMode
		return 0
	}

	fun sceCtrlGetSamplingCycle(cpu: CpuState): Unit = UNIMPLEMENTED(0x02BAAD91)
	fun sceCtrlReadLatch(cpu: CpuState): Unit = UNIMPLEMENTED(0x0B588501)
	fun sceCtrl_348D99D4(cpu: CpuState): Unit = UNIMPLEMENTED(0x348D99D4)
	fun sceCtrlPeekBufferPositive(cpu: CpuState): Unit = UNIMPLEMENTED(0x3A622550)
	fun sceCtrlReadBufferNegative(cpu: CpuState): Unit = UNIMPLEMENTED(0x60B81F86)
	fun sceCtrlSetRapidFire(cpu: CpuState): Unit = UNIMPLEMENTED(0x6841BE1A)
	fun sceCtrlGetIdleCancelThreshold(cpu: CpuState): Unit = UNIMPLEMENTED(0x687660FA)
	fun sceCtrlClearRapidFire(cpu: CpuState): Unit = UNIMPLEMENTED(0xA68FD260)
	fun sceCtrlSetIdleCancelThreshold(cpu: CpuState): Unit = UNIMPLEMENTED(0xA7144800)
	fun sceCtrl_AF5960F3(cpu: CpuState): Unit = UNIMPLEMENTED(0xAF5960F3)
	fun sceCtrlPeekLatch(cpu: CpuState): Unit = UNIMPLEMENTED(0xB1D0E5CD)
	fun sceCtrlPeekBufferNegative(cpu: CpuState): Unit = UNIMPLEMENTED(0xC152080A)
	fun sceCtrlGetSamplingMode(cpu: CpuState): Unit = UNIMPLEMENTED(0xDA6B76A1)


	override fun registerModule() {
		registerFunctionInt("sceCtrlPeekBufferPositive", 0x3A622550, 150, syscall = 0x2150) { sceCtrlPeekBufferPositive(ptr, int) }
		registerFunctionInt("sceCtrlReadBufferPositive", 0x1F803938, since = 150) { sceCtrlReadBufferPositive(ptr, int) }
		registerFunctionInt("sceCtrlSetSamplingCycle", 0x6A2774F3, since = 150) { sceCtrlSetSamplingCycle(int) }
		registerFunctionInt("sceCtrlSetSamplingMode", 0x1F4011E6, since = 150) { sceCtrlSetSamplingMode(int) }

		registerFunctionRaw("sceCtrlGetSamplingCycle", 0x02BAAD91, since = 150) { sceCtrlGetSamplingCycle(it) }
		registerFunctionRaw("sceCtrlReadLatch", 0x0B588501, since = 150) { sceCtrlReadLatch(it) }
		registerFunctionRaw("sceCtrl_348D99D4", 0x348D99D4, since = 150) { sceCtrl_348D99D4(it) }
		registerFunctionRaw("sceCtrlReadBufferNegative", 0x60B81F86, since = 150) { sceCtrlReadBufferNegative(it) }
		registerFunctionRaw("sceCtrlSetRapidFire", 0x6841BE1A, since = 150) { sceCtrlSetRapidFire(it) }
		registerFunctionRaw("sceCtrlGetIdleCancelThreshold", 0x687660FA, since = 150) { sceCtrlGetIdleCancelThreshold(it) }
		registerFunctionRaw("sceCtrlClearRapidFire", 0xA68FD260, since = 150) { sceCtrlClearRapidFire(it) }
		registerFunctionRaw("sceCtrlSetIdleCancelThreshold", 0xA7144800, since = 150) { sceCtrlSetIdleCancelThreshold(it) }
		registerFunctionRaw("sceCtrl_AF5960F3", 0xAF5960F3, since = 150) { sceCtrl_AF5960F3(it) }
		registerFunctionRaw("sceCtrlPeekLatch", 0xB1D0E5CD, since = 150) { sceCtrlPeekLatch(it) }
		registerFunctionRaw("sceCtrlPeekBufferNegative", 0xC152080A, since = 150) { sceCtrlPeekBufferNegative(it) }
		registerFunctionRaw("sceCtrlGetSamplingMode", 0xDA6B76A1, since = 150) { sceCtrlGetSamplingMode(it) }
	}
}
