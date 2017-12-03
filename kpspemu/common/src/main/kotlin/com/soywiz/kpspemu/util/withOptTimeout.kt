package com.soywiz.kpspemu.util

import com.soywiz.korio.CancellationException
import com.soywiz.korio.async.eventLoop
import com.soywiz.korio.async.suspendCancellableCoroutine
import com.soywiz.korio.coroutine.Continuation
import com.soywiz.korio.coroutine.korioStartCoroutine

// @TODO: Move this to Korio
suspend fun <T> withOptTimeout(ms: Int?, name: String = "timeout", callback: suspend () -> T): T = suspendCancellableCoroutine<T> { c ->
	var cancelled = false
	val timer = when {
		ms == null -> null
		ms >= 0 -> c.eventLoop.setTimeout(ms) { c.cancel(CancellationException("")) }
		else -> null
	}
	c.onCancel {
		cancelled = true
		timer?.close()
		c.cancel()
	}
	callback.korioStartCoroutine(object : Continuation<T> {
		override val context = c.context

		override fun resume(value: T) {
			if (cancelled) return
			timer?.close()
			c.resume(value)
		}

		override fun resumeWithException(exception: Throwable) {
			if (cancelled) return
			timer?.close()
			c.resumeWithException(exception)
		}
	})
}
