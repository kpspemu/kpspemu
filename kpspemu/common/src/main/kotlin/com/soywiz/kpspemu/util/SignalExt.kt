package com.soywiz.kpspemu.util

import com.soywiz.korio.async.Promise
import com.soywiz.korio.async.Signal
import com.soywiz.korio.lang.Closeable

fun <T> Signal<T>.waitOnePromise(): Promise<T> {
	val deferred = Promise.Deferred<T>()
	var close: Closeable? = null
	close = once {
		close?.close()
		deferred.resolve(it)
	}
	deferred.onCancel {
		close.close()
	}
	return deferred.promise
}