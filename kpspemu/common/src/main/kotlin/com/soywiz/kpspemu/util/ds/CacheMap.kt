package com.soywiz.kpspemu.util.ds

class CacheMap<K, V>(val maxSize: Int = 16) {
	val entries = LinkedHashMap<K, V>()

	val size: Int get() = entries.size
	fun has(key: K) = entries.containsKey(key)
	operator fun get(key: K) = entries[key]
	operator fun set(key: K, value: V) {
		if (size >= maxSize && !entries.containsKey(key)) {
			entries.remove(entries.keys.first())
		}

		entries.remove(key) // refresh if exists
		entries[key] = value
	}

	inline fun getOrPut(key: K, callback: (K) -> V): V {
		if (!has(key)) set(key, callback(key))
		return get(key)!!
	}
}