package com.soywiz.kpspemu.hle.modules

import com.soywiz.kpspemu.Emulator
import com.soywiz.kpspemu.cpu.CpuState
import com.soywiz.kpspemu.hle.SceModule

@Suppress("ClassName", "UNUSED_PARAMETER")
class sceUmdUser(emulator: Emulator) : SceModule(emulator, "sceUmdUser", 0x40010011, "np9660.prx", "sceNp9660_driver") {
	val PSP_UMD_INIT = 0x00
	val PSP_UMD_NOT_PRESENT = 0x01
	val PSP_UMD_PRESENT = 0x02
	val PSP_UMD_CHANGED = 0x04
	val PSP_UMD_NOT_READY = 0x08
	val PSP_UMD_READY = 0x10
	val PSP_UMD_READABLE = 0x20

	fun sceUmdCheckMedium(): Int {
		return 1 // Inserted
	}

	fun sceUmdActivate(unit: Int, device: String): Int {
		return 0
	}

	fun sceUmdWaitDriveStat(stat: Int): Int {
		return 0
	}

	fun sceUmdWaitDriveStatCB(stat: Int, timeout: Int): Int {
		return 0
	}

	fun sceUmdGetDriveStat(): Int = PSP_UMD_PRESENT or PSP_UMD_READY or PSP_UMD_READABLE

	fun sceUmdRegisterUMDCallBack(callbackId: Int): Int {
		logger.error { "sceUmdRegisterUMDCallBack UNIMPLEMENTED" }
		return 0
	}
	fun sceUmdUnRegisterUMDCallBack(callbackId: Int): Int {
		logger.error { "sceUmdUnRegisterUMDCallBack UNIMPLEMENTED" }
		return 0
	}

	fun sceUmdGetErrorStat(cpu: CpuState): Unit = UNIMPLEMENTED(0x20628E6F)
	fun sceUmdGetDiscInfo(cpu: CpuState): Unit = UNIMPLEMENTED(0x340B7686)
	fun sceUmdWaitDriveStatWithTimer(cpu: CpuState): Unit = UNIMPLEMENTED(0x56202973)
	fun sceUmdCancelWaitDriveStat(cpu: CpuState): Unit = UNIMPLEMENTED(0x6AF9B50A)
	fun sceUmdReplaceProhibit(cpu: CpuState): Unit = UNIMPLEMENTED(0x87533940)
	fun sceUmdReplacePermit(cpu: CpuState): Unit = UNIMPLEMENTED(0xCBE9F02A)
	fun sceUmdDeactivate(cpu: CpuState): Unit = UNIMPLEMENTED(0xE83742BA)

	override fun registerModule() {
		registerFunctionInt("sceUmdCheckMedium", 0x46EBB729, since = 150) { sceUmdCheckMedium() }
		registerFunctionInt("sceUmdActivate", 0xC6183D47, since = 150) { sceUmdActivate(int, istr) }
		registerFunctionInt("sceUmdWaitDriveStat", 0x8EF08FCE, since = 150) { sceUmdWaitDriveStat(int) }
		registerFunctionInt("sceUmdWaitDriveStatCB", 0x4A9E5E29, since = 150) { sceUmdWaitDriveStatCB(int, int) }
		registerFunctionInt("sceUmdGetDriveStat", 0x6B4A146C, since = 150) { sceUmdGetDriveStat() }
		registerFunctionInt("sceUmdRegisterUMDCallBack", 0xAEE7404D, since = 150) { sceUmdRegisterUMDCallBack(int) }
		registerFunctionInt("sceUmdUnRegisterUMDCallBack", 0xBD2BDE07, since = 150) { sceUmdUnRegisterUMDCallBack(int) }

		registerFunctionRaw("sceUmdGetErrorStat", 0x20628E6F, since = 150) { sceUmdGetErrorStat(it) }
		registerFunctionRaw("sceUmdGetDiscInfo", 0x340B7686, since = 150) { sceUmdGetDiscInfo(it) }
		registerFunctionRaw("sceUmdWaitDriveStatWithTimer", 0x56202973, since = 150) { sceUmdWaitDriveStatWithTimer(it) }
		registerFunctionRaw("sceUmdCancelWaitDriveStat", 0x6AF9B50A, since = 150) { sceUmdCancelWaitDriveStat(it) }
		registerFunctionRaw("sceUmdReplaceProhibit", 0x87533940, since = 150) { sceUmdReplaceProhibit(it) }
		registerFunctionRaw("sceUmdReplacePermit", 0xCBE9F02A, since = 150) { sceUmdReplacePermit(it) }
		registerFunctionRaw("sceUmdDeactivate", 0xE83742BA, since = 150) { sceUmdDeactivate(it) }
	}
}
