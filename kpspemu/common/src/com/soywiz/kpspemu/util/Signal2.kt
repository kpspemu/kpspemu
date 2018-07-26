package com.soywiz.kpspemu.util

import com.soywiz.kds.*
import com.soywiz.korio.async.*
import com.soywiz.korio.lang.*
import kotlinx.coroutines.*
import kotlin.coroutines.*

class Signal2<T>(val onRegister: () -> Unit = {}) { //: AsyncSequence<T> {
    inner class Node(val once: Boolean, val item: (T) -> Unit) : Closeable {
        override fun close() {
            handlers.remove(this)
        }
    }

    private var handlers = LinkedList<Node>()

    val listenerCount: Int get() = handlers.size

    fun once(handler: (T) -> Unit): Closeable = _add(true, handler)
    fun add(handler: (T) -> Unit): Closeable = _add(false, handler)

    fun clear() = handlers.clear()

    private fun _add(once: Boolean, handler: (T) -> Unit): Closeable {
        onRegister()
        val node = Node(once, handler)
        handlers.add(node)
        return node
    }

    operator fun invoke(value: T) {
        val it = handlers.iterator()
        while (it.hasNext()) {
            val node = it.next()
            if (node.once) it.remove()
            node.item(value)
        }
    }

    operator fun invoke(handler: (T) -> Unit): Closeable = add(handler)
}

fun <TI, TO> Signal2<TI>.mapSignal(transform: (TI) -> TO): Signal2<TO> {
    val out = Signal2<TO>()
    this.add { out(transform(it)) }
    return out
}

operator fun Signal2<Unit>.invoke() = invoke(Unit)

suspend fun <T> Signal2<T>.waitOne(timeout: Int? = null): T = suspendCancellableCoroutine { c ->
    var close: Closeable? = null
    var close2: Job? = null
    fun closeAll() {
        close?.close()
        close2?.cancel()
    }
    close = once {
        closeAll()
        c.resume(it)
    }
    if (timeout != null) {
        close2 = launchImmediately(c.context) {
            delay(timeout)
            closeAll()
            c.resumeWithException(TimeoutException())
        }
    }
    c.invokeOnCancellation {
        closeAll()
    }
}

fun <T> Signal2<T>.waitOnePromise(): Deferred<T> {
    val deferred = CompletableDeferred<T>()
    var close: Closeable? = null
    close = once {
        close?.close()
        deferred.complete(it)
    }
    deferred.invokeOnCompletion {
        if (deferred.isCancelled) {
            close.close()
        }
    }
    return deferred
}
