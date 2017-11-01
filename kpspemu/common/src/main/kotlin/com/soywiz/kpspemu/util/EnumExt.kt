package com.soywiz.kpspemu.util

interface NumericEnum {
	val id: Int
}

interface Flags<T> : NumericEnum {
	infix fun hasFlag(item: Flags<T>): Boolean = (id and item.id) == item.id
}

interface IdEnum {
	val id: Int

	open class SmallCompanion<T : IdEnum>(val values: Array<T>) {
		private val defaultValue: T = values.first()
		private val MAX_ID = values.map { it.id }.max() ?: 0
		private val valuesById = ArrayList<T>()

		init {
			for (n in 0 until MAX_ID + 1) valuesById.add(defaultValue)
			for (v in values) valuesById[v.id] = v
		}

		operator fun invoke(id: Int) = valuesById.getOrElse(id) { defaultValue }
	}
}