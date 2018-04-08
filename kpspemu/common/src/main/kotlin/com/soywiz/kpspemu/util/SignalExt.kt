package com.soywiz.kpspemu.util

import com.soywiz.klock.*
import com.soywiz.korio.async.*
import com.soywiz.korio.lang.*

suspend fun <T> Signal<T>.waitOneTimeout(timeout: TimeSpan): T = suspendCancellableCoroutine { c ->
    var close: Closeable? = null
    val timer = c.eventLoop.setTimeout(timeout.ms) {
        close?.close()
        c.cancel(TimeoutException())
    }
    close = once {
        close?.close()
        timer.close()
        c.resume(it)
    }
    c.onCancel {
        close.close()
        timer.close()
    }
}

suspend fun <T> Signal<T>.waitOneOptTimeout(timeout: TimeSpan? = null): T = when {
    timeout != null -> waitOneTimeout(timeout)
    else -> waitOne()
}
