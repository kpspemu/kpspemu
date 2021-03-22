package com.soywiz.korio.async

import kotlinx.coroutines.*
import kotlin.coroutines.*

suspend fun <T> CoroutineContext.withOptTimeout(timeout: Long?, name: String, block: suspend CoroutineScope.() -> T): T {
    if (timeout == null) {
        return block(CoroutineScope(this))
    } else {
        return kotlinx.coroutines.withTimeout(timeout, block)
    }
}