package com.soywiz.kpspemu.util

interface NumericEnum {
	val id: Int
}

interface Flags<T> : NumericEnum {
	infix fun hasFlag(item: Flags<T>): Boolean = (id and item.id) == item.id
}
