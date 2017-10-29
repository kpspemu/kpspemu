package com.soywiz.kpspemu.hle

import com.soywiz.kpspemu.Emulator
import com.soywiz.kpspemu.hle.manager.ModuleManager
import com.soywiz.kpspemu.hle.modules.*

fun Emulator.registerNativeModules() = moduleManager.registerNativeModules()

fun ModuleManager.registerNativeModules() {
	register(LoadExecForUser(emulator))
	register(ThreadManForUser(emulator))
	register(SysMemUserForUser(emulator))
	register(StdioForUser(emulator))
	register(ModuleMgrForUser(emulator))
	register(IoFileMgrForUser(emulator))
	register(sceGe_user(emulator))
	register(sceRtc(emulator))
	register(sceCtrl(emulator))
	register(sceDisplay(emulator))
	register(UtilsForUser(emulator))
}