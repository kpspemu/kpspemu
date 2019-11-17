package com.soywiz.korio.error

val invalidOp: Nothing get() = com.soywiz.korio.lang.invalidOp
fun invalidOp(str: String): Nothing = com.soywiz.korio.lang.invalidOp(str)