package com.soywiz.korau.format.util

import com.soywiz.korio.lang.format
import com.soywiz.korio.util.signExtend

@Deprecated("", ReplaceWith("format"))
fun String_format(format: String): String = format

@Deprecated("", ReplaceWith("format.format(*args)", "com.soywiz.korio.lang.format"))
fun String_format(format: String, vararg args: Any): String = format.format(*args)

@Deprecated("", ReplaceWith("v.signExtend(bits)", "com.soywiz.korio.util.signExtend"))
fun signExtend(v: Int, bits: Int) = v.signExtend(bits)