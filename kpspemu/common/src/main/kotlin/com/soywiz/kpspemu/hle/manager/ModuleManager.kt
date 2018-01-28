package com.soywiz.kpspemu.hle.manager

import com.soywiz.korio.error.invalidOp
import com.soywiz.kpspemu.Emulator
import com.soywiz.kpspemu.hle.SceModule
import kotlin.reflect.KClass

class ModuleManager(val emulator: Emulator) {
    val modules = LinkedHashMap<String, SceModule>()
    val modulesByClass = LinkedHashMap<KClass<*>, SceModule>()

    fun register(module: SceModule) {
        modules[module.name] = module
        modulesByClass[module::class] = module
        module.registerPspModule()
    }

    fun getByName(name: String): SceModule = modules[name] ?: invalidOp("Can't find module '$name'")

    fun reset() {
        modules.clear()
    }
}