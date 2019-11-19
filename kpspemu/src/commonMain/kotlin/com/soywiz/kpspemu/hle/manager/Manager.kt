package com.soywiz.kpspemu.hle.manager

import com.soywiz.kds.*
import com.soywiz.kpspemu.*

class ResourceNotFoundException(msg: String) : Exception(msg)

open class Manager<T : Resource>(val name: String, override val emulator: Emulator, val initialId: Int = 0) : WithEmulator {
    internal var lastId: Int = initialId
    internal var freeIds = Pool { lastId++ }
    internal val resourcesById = LinkedHashMap<Int, T>()
    val resourcesCount: Int get() = resourcesById.size

    open fun put(item: T): T = item.apply { resourcesById[item.id] = item }
    internal fun allocId(): Int = freeIds.alloc()
    fun tryGetByName(name: String): T? = resourcesById.values.firstOrNull { it.name == name }
    fun tryGetById(id: Int): T? = resourcesById[id]
    fun getById(id: Int) = tryGetById(id) ?: throw ResourceNotFoundException("Can't find element $id in $name")
    open fun freeById(id: Int) {
        freeIds.free(id)
        resourcesById.remove(id)
    }

    open fun reset() {
        lastId = initialId
        freeIds = Pool { lastId++ }
        resourcesById.clear()
    }
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