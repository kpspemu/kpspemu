package com.soywiz.kpspemu.util

inline fun <T, R> Iterable<T>.reduceAcumulate(initial: R, callback: (prev: R, item: T) -> R): R {
    var value = initial
    for (it in this) {
        value = callback(value, it)
    }
    return value
}