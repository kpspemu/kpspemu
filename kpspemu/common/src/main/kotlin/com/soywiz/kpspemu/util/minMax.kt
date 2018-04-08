package com.soywiz.kpspemu.util

import kotlin.math.*

fun max(a: Float, b: Float, c: Float) = max(max(a, b), c)
fun max(a: Float, b: Float, c: Float, d: Float) = max(max(max(a, b), c), d)
fun max(a: Float, b: Float, c: Float, d: Float, e: Float) = max(max(max(max(a, b), c), d), e)

fun max(a: Int, b: Int, c: Int) = max(max(a, b), c)
fun max(a: Int, b: Int, c: Int, d: Int) = max(max(max(a, b), c), d)
fun max(a: Int, b: Int, c: Int, d: Int, e: Int) = max(max(max(max(a, b), c), d), e)