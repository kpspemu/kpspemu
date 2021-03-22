package com.soywiz.kpspemu.hle.manager

import com.soywiz.korio.error.*
import com.soywiz.kpspemu.*
import com.soywiz.kpspemu.hle.*
import kotlin.reflect.*

class ModuleManager(val emulator: Emulator) {
    val modules = LinkedHashMap<String, SceModule>()
    val modulesByClass = LinkedHashMap<KClass<*>, SceModule>()

    fun register(module: SceModule) {
        modules[module.name] = module
        modulesByClass[module::class] = module
        module.registerPspModule()
    }

    fun getByName(name: String): SceModule = modules[name] ?: invalidOp("Can't find module '$name'")
    inline fun <reified T : SceModule> get(): SceModule = modulesByClass[T::class] ?: invalidOp("Can't find module '${T::class}'")

    fun reset() {
        for (module in modules.values) {
            module.stopModule()
        }
        modules.clear()
    }
}