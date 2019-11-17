package com.soywiz.kpspemu.util

inline fun <T> ignoreErrors(callback: () -> T): T? = try {
    callback()
} catch (e: Throwable) {
    null
}