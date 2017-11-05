package com.soywiz.kpspemu.util

actual class IntMap<T> actual constructor() {
	// @TODO: Optimize
	val map = HashMap<Int, T>()

	actual fun remove(key: Int): Unit = run { map.remove(key) }
	actual operator fun get(key: Int): T? = map[key]
	actual operator fun set(key: Int, value: T): Unit = run { map[key] = value }
}