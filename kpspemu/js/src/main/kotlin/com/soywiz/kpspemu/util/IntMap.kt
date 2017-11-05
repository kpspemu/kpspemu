package com.soywiz.kpspemu.util

external class Map<T> {
	fun delete(key: dynamic): Unit
	fun set(key: dynamic, value: T): T
	fun get(key: dynamic): T?
}

actual class IntMap<T> actual constructor() {
	val map = Map<T>()
	actual fun remove(key: Int): Unit = map.delete(key)
	actual operator fun get(key: Int): T? = map.get(key)
	actual operator fun set(key: Int, value: T): Unit {
		map.set(key, value)
	}
}
