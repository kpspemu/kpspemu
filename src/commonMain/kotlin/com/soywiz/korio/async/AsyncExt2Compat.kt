package com.soywiz.korio.async

import com.soywiz.klock.*
import com.soywiz.korio.lang.*
import kotlin.coroutines.*

suspend fun <T> Signal<T>.waitOneFixed(timeout: TimeSpan): T? = kotlinx.coroutines.suspendCancellableCoroutine { c ->
    var close: Closeable? = null
    var running = true

    fun closeAll() {
        running = false
        close?.close()
        close = null
    }

    launchImmediately(c.context) {
        delay(timeout)
        if (running) {
            closeAll()
            c.resume(null)
        }
    }

    close = once {
        closeAll()
        c.resume(it)
    }

    c.invokeOnCancellation {
        closeAll()
    }
}