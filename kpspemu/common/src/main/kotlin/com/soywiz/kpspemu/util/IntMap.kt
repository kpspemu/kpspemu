package com.soywiz.kpspemu.util

expect class IntMap<T>() {
	fun remove(key: Int): Unit
	operator fun get(key: Int): T?
	operator fun set(key: Int, value: T): Unit
}

fun <T> IntMap<T>.getOrPut(key: Int, callback: () -> T): T {
	val res = get(key)
	if (res == null) set(key, callback())
	return get(key)!!
}