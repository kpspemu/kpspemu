package com.soywiz.kpspemu.util

class FloatArray2(val cols: Int, val rows: Int, val data: FloatArray = FloatArray(rows * cols)) {
	fun index(col: Int, row: Int): Int = row * cols + col
	operator fun get(col: Int, row: Int) = data[index(col, row)]
	operator fun set(col: Int, row: Int, value: Float) = run { data[index(col, row)] = value }

	//operator fun get(index: Int) = data[index]
	//operator fun set(index: Int, value: Float) = run { data[index] = value }
}