package com.soywiz.kpspemu.hle.modules

import com.soywiz.kpspemu.Emulator
import com.soywiz.kpspemu.cpu.CpuState
import com.soywiz.kpspemu.hle.SceModule
import com.soywiz.kpspemu.mem.Ptr

@Suppress("UNUSED_PARAMETER")
class sceImpose(emulator: Emulator) : SceModule(emulator, "sceImpose", 0x40010011, "impose.prx", "sceImpose_Driver") {
	fun sceImposeGetBatteryIconStatus(charging: Ptr, status: Ptr): Int {
		charging.sw(0, emulator.battery.chargingType.id)
		status.sw(0, emulator.battery.iconStatus.id)
		return 0
	}

	fun sceImposeGetHomePopup(cpu: CpuState): Unit = UNIMPLEMENTED(0x0F341BE4)
	fun sceImposeGetLanguageMode(cpu: CpuState): Unit = UNIMPLEMENTED(0x24FD7BCF)
	fun sceImposeSetLanguageMode(cpu: CpuState): Unit = UNIMPLEMENTED(0x36AA6E91)
	fun sceImposeHomeButton(cpu: CpuState): Unit = UNIMPLEMENTED(0x381BD9E7)
	fun sceImposeSetHomePopup(cpu: CpuState): Unit = UNIMPLEMENTED(0x5595A71A)
	fun sceImposeSetUMDPopup(cpu: CpuState): Unit = UNIMPLEMENTED(0x72189C48)
	fun sceImposeGetBacklightOffTime(cpu: CpuState): Unit = UNIMPLEMENTED(0x8F6E3518)
	fun sceImposeSetBacklightOffTime(cpu: CpuState): Unit = UNIMPLEMENTED(0x967F6D4A)
	fun sceImpose_9BA61B49(cpu: CpuState): Unit = UNIMPLEMENTED(0x9BA61B49)
	fun sceImpose_A9884B00(cpu: CpuState): Unit = UNIMPLEMENTED(0xA9884B00)
	fun sceImpose_BB3F5DEC(cpu: CpuState): Unit = UNIMPLEMENTED(0xBB3F5DEC)
	fun sceImposeGetUMDPopup(cpu: CpuState): Unit = UNIMPLEMENTED(0xE0887BC8)
	fun sceImpose_FCD44963(cpu: CpuState): Unit = UNIMPLEMENTED(0xFCD44963)
	fun sceImpose_FF1A2F07(cpu: CpuState): Unit = UNIMPLEMENTED(0xFF1A2F07)


	override fun registerModule() {
		registerFunctionInt("sceImposeGetBatteryIconStatus", 0x8C943191, since = 150) { sceImposeGetBatteryIconStatus(ptr, ptr) }

		registerFunctionRaw("sceImposeGetHomePopup", 0x0F341BE4, since = 150) { sceImposeGetHomePopup(it) }
		registerFunctionRaw("sceImposeGetLanguageMode", 0x24FD7BCF, since = 150) { sceImposeGetLanguageMode(it) }
		registerFunctionRaw("sceImposeSetLanguageMode", 0x36AA6E91, since = 150) { sceImposeSetLanguageMode(it) }
		registerFunctionRaw("sceImposeHomeButton", 0x381BD9E7, since = 150) { sceImposeHomeButton(it) }
		registerFunctionRaw("sceImposeSetHomePopup", 0x5595A71A, since = 150) { sceImposeSetHomePopup(it) }
		registerFunctionRaw("sceImposeSetUMDPopup", 0x72189C48, since = 150) { sceImposeSetUMDPopup(it) }
		registerFunctionRaw("sceImposeGetBacklightOffTime", 0x8F6E3518, since = 150) { sceImposeGetBacklightOffTime(it) }
		registerFunctionRaw("sceImposeSetBacklightOffTime", 0x967F6D4A, since = 150) { sceImposeSetBacklightOffTime(it) }
		registerFunctionRaw("sceImpose_9BA61B49", 0x9BA61B49, since = 150) { sceImpose_9BA61B49(it) }
		registerFunctionRaw("sceImpose_A9884B00", 0xA9884B00, since = 150) { sceImpose_A9884B00(it) }
		registerFunctionRaw("sceImpose_BB3F5DEC", 0xBB3F5DEC, since = 150) { sceImpose_BB3F5DEC(it) }
		registerFunctionRaw("sceImposeGetUMDPopup", 0xE0887BC8, since = 150) { sceImposeGetUMDPopup(it) }
		registerFunctionRaw("sceImpose_FCD44963", 0xFCD44963, since = 150) { sceImpose_FCD44963(it) }
		registerFunctionRaw("sceImpose_FF1A2F07", 0xFF1A2F07, since = 150) { sceImpose_FF1A2F07(it) }
	}
}
