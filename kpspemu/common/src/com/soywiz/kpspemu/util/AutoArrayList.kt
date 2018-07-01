package com.soywiz.kpspemu.util

class AutoArrayList<T>(val gen: (Int) -> T) {
    val items = arrayListOf<T>()

    operator fun get(index: Int): T {
        while (items.size <= index) items += gen(items.size)
        return items[index]
    }
}