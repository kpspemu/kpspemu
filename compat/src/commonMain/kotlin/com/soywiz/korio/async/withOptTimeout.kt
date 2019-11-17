package com.soywiz.korio.async

import kotlinx.coroutines.*

suspend fun <T> CoroutineScope.withOptTimeout(timeout: Long?, name: String, block: suspend CoroutineScope.() -> T): T {
    if (timeout == null) {
        return block()
    } else {
        return kotlinx.coroutines.withTimeout(timeout, block)
    }
}