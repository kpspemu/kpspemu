package com.soywiz.korio.async

import kotlinx.coroutines.channels.*

class AsyncPoolCompat<T>(val maxItems: Int = Int.MAX_VALUE, val create: suspend () -> T) {
    var createdItems = 0
    private val freedItem = Channel<T>(maxItems)

    suspend fun alloc(): T {
        return if (createdItems >= maxItems) {
            freedItem.receive()
        } else {
            createdItems++
            create()
        }
    }

    fun free(item: T) {
        freedItem.trySend(item)
    }

    suspend inline fun <TR> alloc(callback: suspend (T) -> TR): TR {
        val item = alloc()
        try {
            return callback(item)
        } finally {
            free(item)
        }
    }
}
