package com.soywiz.kpspemu.hle.manager

import com.soywiz.korio.error.invalidOp
import com.soywiz.kpspemu.Emulator
import com.soywiz.kpspemu.hle.modules.SceModule

class ModuleManager(val e: Emulator) {
	val modules = LinkedHashMap<String, SceModule>()

	fun register(module: SceModule) {
		modules[module.name] = module
		module.registerPspModule(e)
	}

	fun getByName(name: String): SceModule = modules[name] ?: invalidOp("Can't find module '$name'")
}