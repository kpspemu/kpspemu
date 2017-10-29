package com.soywiz.kpspemu.util

class IntMap<T> {
	// @TODO: Optimize
	val map = LinkedHashMap<Int, T>()

	fun remove(key: Int): T? = map.remove(key)
	operator fun get(key: Int): T? = map[key]
	operator fun set(key: Int, value: T): Unit = run { map[key] = value }
}