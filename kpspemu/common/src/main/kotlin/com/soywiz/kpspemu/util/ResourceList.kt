package com.soywiz.kpspemu.util

import com.soywiz.kds.IntMap
import com.soywiz.kds.Pool
import com.soywiz.korio.error.invalidOp
import com.soywiz.korio.util.hex

interface ResourceItem {
	val id: Int
}

class ResourceList<T : ResourceItem>(val name: String, private val create: (id: Int) -> T) {
	private val items = IntMap<T>()
	private var lastId: Int = 1
	private val freeList = Pool<T>() { create(lastId++) }

	fun alloc(): T {
		val item = freeList.alloc()
		items[item.id] = item
		return item
	}

	fun free(item: T) {
		freeList.free(item)
		items.remove(item.id)
	}

	fun freeById(id: Int) {
		if (id in this) {
			free(this[id])
		}
	}

	fun tryGetById(id: Int): T? = items[id]
	operator fun get(id: Int): T = tryGetById(id) ?: invalidOp("Can't find $name with id ${id.hex}")
	operator fun contains(id: Int): Boolean = tryGetById(id) != null
}