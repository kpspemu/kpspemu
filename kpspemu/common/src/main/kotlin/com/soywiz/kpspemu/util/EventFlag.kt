package com.soywiz.kpspemu.util

import com.soywiz.korio.async.*
import kotlin.reflect.*

class EventFlag<T>(val initial: T) {
    var value: T = initial
        set(value) {
            if (field != value) {
                field = value
                onUpdated(Unit)
            }
        }
    val onUpdated = Signal<Unit>()

    suspend fun waitValue(expected: T) {
        while (value != expected) onUpdated.waitOne()
    }

    operator fun getValue(obj: Any?, property: KProperty<*>): T = value
    operator fun setValue(obj: Any?, property: KProperty<*>, value: T): Unit = run { this.value = value }
}

suspend fun EventFlag<Int>.waitAllBits(expected: Int) {
    while ((value and expected) != expected) onUpdated.waitOne()
}

suspend fun EventFlag<Int>.waitAnyBits(expected: Int) {
    while ((value and expected) == 0) onUpdated.waitOne()
}

class EventStatus(val generator: () -> Int) {
    private val onUpdated = Signal<Unit>()
    val v: Int get() = generator()
    operator fun getValue(obj: Any?, property: KProperty<*>): Int = generator()

    fun updated() = onUpdated(Unit)

    suspend fun waitValue(expected: Int) {
        while (v != expected) onUpdated.waitOne()
    }

    suspend fun waitAllBits(expected: Int) {
        while ((v and expected) != expected) onUpdated.waitOne()
    }

    suspend fun waitAnyBits(expected: Int) {
        while ((v and expected) == 0) onUpdated.waitOne()
    }
}
