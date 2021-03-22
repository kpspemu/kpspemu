package com.soywiz.kpspemu.util

import com.soywiz.kds.*
import com.soywiz.korio.error.*
import com.soywiz.krypto.encoding.*

interface ResourceItem {
    val id: Int
}

class ResourceList<T : ResourceItem>(
    val name: String,
    private val notFound: (id: Int) -> Nothing = { invalidOp("Can't find $name with id ${it.hex}") },
    private val create: (id: Int) -> T
) {
    private var items = IntMap<T>()
    private var lastId: Int = 1
    private var freeList = Pool<T>() { create(lastId++) }

    fun alloc(): T {
        val item = freeList.alloc()
        items[item.id] = item
        return item
    }

    fun free(item: T) {
        freeList.free(item)
        items.remove(item.id)
    }

    fun freeById(id: Int): Boolean {
        if (id in this) {
            free(this[id])
            return true
        } else {
            return false
        }
    }

    fun tryGetById(id: Int): T? = items[id]
    operator fun get(id: Int): T = tryGetById(id) ?: notFound(id)
    operator fun contains(id: Int): Boolean = tryGetById(id) != null
    fun reset() {
        lastId = 1
        items = IntMap()
        freeList = Pool<T>() { create(lastId++) }
    }
}