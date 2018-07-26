package com.soywiz.kpspemu.util

import com.soywiz.klock.*
import com.soywiz.korio.async.*
import com.soywiz.korio.lang.*
import kotlinx.coroutines.*
import kotlin.coroutines.*

suspend fun <T> Signal<T>.waitOneTimeout(timeout: TimeSpan): T = suspendCancellableCoroutine { c ->
    var close: Closeable? = null

    val timer = asyncImmediately(c.context) {
        c.context.delay(timeout)
        close?.close()
        c.cancel(TimeoutException())
    }
    close = once {
        close?.close()
        timer.cancel()
        c.resume(it)
    }
    c.invokeOnCancellation {
        close.close()
        timer.cancel()
    }
}

suspend fun <T> Signal<T>.waitOneOptTimeout(timeout: TimeSpan? = null): T = when {
    timeout != null -> waitOneTimeout(timeout)
    else -> waitOne()
}
