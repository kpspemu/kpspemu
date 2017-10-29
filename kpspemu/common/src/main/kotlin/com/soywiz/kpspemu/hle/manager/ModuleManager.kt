package com.soywiz.kpspemu.hle.manager

import com.soywiz.korio.ds.lmapOf
import com.soywiz.kpspemu.hle.modules.SceModule

class ModuleManager {
	val modules = lmapOf<String, () -> SceModule>()

	fun register(name: String, callback: () -> SceModule) {
		modules[name] = callback
	}
}