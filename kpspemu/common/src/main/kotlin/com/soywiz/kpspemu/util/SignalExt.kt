package com.soywiz.kpspemu.util

import com.soywiz.klock.TimeSpan
import com.soywiz.korio.async.Signal
import com.soywiz.korio.async.eventLoop
import com.soywiz.korio.async.suspendCancellableCoroutine
import com.soywiz.korio.lang.Closeable

suspend fun <T> Signal<T>.waitOneTimeout(time: TimeSpan): T = suspendCancellableCoroutine { c ->
	var close: Closeable? = null
	val timer = c.eventLoop.setTimeout(time.ms) {
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
