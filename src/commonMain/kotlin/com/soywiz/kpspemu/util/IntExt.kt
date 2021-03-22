package com.soywiz.kpspemu.util

// Useful function
fun Int.asNull(nullValue: Int = 0): Int? = if (this == nullValue) null else this
