package com.soywiz.kpspemu.hle.modules

import com.soywiz.kpspemu.hle.manager.ModuleManager

fun ModuleManager.registerNativeModules() {
	register("sceCtrl") { sceCtrl() }
	register("sceDisplay") { sceDisplay() }
	register("UtilsForUser") { UtilsForUser() }
}