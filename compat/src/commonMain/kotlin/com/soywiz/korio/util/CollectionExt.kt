package com.soywiz.korio.util

public inline fun <T, R> Iterable<T>.firstNotNullOrNull(predicate: (T) -> R?): R? {
    for (e in this) {
        val res = predicate(e)
        if (res != null) return res
    }
    return null
}

inline fun <T, R> Iterable<T>.reduceAcumulate(initial: R, callback: (prev: R, item: T) -> R): R {
    var value = initial
    for (it in this) {
        value = callback(value, it)
    }
    return value
}
