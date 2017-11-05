package com.soywiz.kpspemu.hle.manager

import com.soywiz.korio.error.invalidOp
import com.soywiz.korio.util.Pool
import com.soywiz.kpspemu.Emulator
import com.soywiz.kpspemu.WithEmulator

open class Manager<T : Resource>(val name: String, override val emulator: Emulator) : WithEmulator {
	internal var lastId: Int = 0
	internal val freeIds = Pool { lastId++ }
	internal val resourcesById = LinkedHashMap<Int, T>()
	val resourcesCount: Int get() = resourcesById.size

	internal fun allocId(): Int = freeIds.alloc()
	fun tryGetByName(name: String): T? = resourcesById.values.firstOrNull { it.name == name }
	fun tryGetById(id: Int): T? = resourcesById[id]
	fun getById(id: Int) = tryGetById(id) ?: invalidOp("Can't find $name $id")
}

open class Resource(
	val manager: Manager<out Resource>,
	val id: Int,
	val name: String
) : WithEmulator {
	override val emulator get() = manager.emulator

	init {
		(manager.resourcesById as LinkedHashMap<Int, Resource>).set(id, this)
	}

	fun free() {
		manager.freeIds.free(id)
		manager.resourcesById.remove(id)
	}
}