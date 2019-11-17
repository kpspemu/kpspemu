package com.soywiz.kpspemu.util

import com.soywiz.korio.async.*
//import kotlinx.atomicfu.*

interface Resetable {
    fun reset(): Unit
}

interface PoolItem : Resetable {
    val id: Int
}

/*
class AsyncPool2<T : PoolItem>(val maxItems: Int = Int.MAX_VALUE, var initId: Int = 0, val create: suspend (Int) -> T) {
    var createdItems = atomic(0)
    private val freedItem = ProduceConsumer<T>()
    val allocatedItems = LinkedHashMap<Int, T>()

    operator fun get(id: Int): T? = allocatedItems[id]

    suspend fun <TR> tempAlloc(callback: suspend (T) -> TR): TR {
        val item = alloc()
        try {
            return callback(item)
        } finally {
            free(item)
        }
    }

    suspend fun alloc(): T {
        val res = if (createdItems.value >= maxItems) {
            freedItem.consume()!!
        } else {
            createdItems.addAndGet(1)
            create(initId++)
        }
        res.reset()
        allocatedItems[res.id] = res
        return res
    }

    fun free(item: T) {
        freedItem.produce(item)
        allocatedItems.remove(item.id)
    }
}
s*/
