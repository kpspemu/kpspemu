package com.soywiz.korge.util

fun <T> MutableList<T>.splice(index: Int, count: Int, vararg values: T) {
    val temp = this.toList()
    this.clear()
    for (n in 0 until index) {
        this.add(temp[n])
    }
    for (value in values) this.add(value)
    for (n in index + count until temp.size) {
        this.add(temp[n])
    }
}
