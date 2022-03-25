package com.soywiz.kpspemu.util

import com.soywiz.korio.util.*

// @TODO: Update korio with this: Allows to handle negative values!
open class SmallCompanion2<T : IdEnum>(val values: Array<T>) {
    private val defaultValue: T = values.first()
    private val MIN_ID = values.minOfOrNull { it.id } ?: 0
    private val MAX_ID = values.maxOfOrNull { it.id } ?: 0
    private val SIZE_ID = MAX_ID - MIN_ID
    private val valuesById = Array<Any>(SIZE_ID + 1) { defaultValue }

    init {
        for (v in values) valuesById[v.id - MIN_ID] = v
    }

    operator fun invoke(id: Int): T = valuesById.getOrElse(id - MIN_ID) { defaultValue } as T
}
