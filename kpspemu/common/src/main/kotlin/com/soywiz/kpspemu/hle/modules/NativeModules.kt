package com.soywiz.kpspemu.hle.modules

import com.soywiz.kpspemu.Emulator
import com.soywiz.kpspemu.hle.manager.ModuleManager

fun Emulator.registerNativeModules() = moduleManager.registerNativeModules()

fun ModuleManager.registerNativeModules() {
	register(sceCtrl())
	register(sceDisplay())
	register(UtilsForUser())
}