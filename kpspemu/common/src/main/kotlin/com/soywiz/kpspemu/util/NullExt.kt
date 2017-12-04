package com.soywiz.kpspemu.util

inline fun <T> T.nullIf(callback: T.() -> Boolean): T? = if (callback(this)) null else this