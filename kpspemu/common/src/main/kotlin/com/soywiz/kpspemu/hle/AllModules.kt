package com.soywiz.kpspemu.hle

import com.soywiz.kpspemu.Emulator
import com.soywiz.kpspemu.hle.manager.ModuleManager
import com.soywiz.kpspemu.hle.modules.*

fun Emulator.registerNativeModules() = moduleManager.registerNativeModules()

fun ModuleManager.registerNativeModules() {
	register(LoadExecForUser())
	register(ThreadManForUser())
	register(SysMemUserForUser())
	register(StdioForUser())
	register(ModuleMgrForUser())
	register(IoFileMgrForUser())
	register(sceGe_user())
	register(sceRtc())
	register(sceCtrl())
	register(sceDisplay())
	register(UtilsForUser())
}