package com.soywiz.kpspemu.util

import com.soywiz.korio.async.Signal
import com.soywiz.korio.async.waitOne
import kotlin.reflect.KProperty

class EventFlag<T>(val initial: T) {
	var value: T = initial
		set(value) {
			if (field != value) {
				field = value
				onUpdated(Unit)
			}
		}
	val onUpdated = Signal<Unit>()

	suspend fun waitFor(expected: T) {
		while (value != expected) {
			onUpdated.waitOne()
		}
	}

	operator fun getValue(obj: Any?, property: KProperty<*>): T = value
	operator fun setValue(obj: Any?, property: KProperty<*>, value: T): Unit = run { this.value = value }
}