package com.soywiz.kpspemu.util

import com.soywiz.klock.Klock

fun Klock.currentTimeMicroDouble() = millisecondsToMicroseconds(currentTimeMillisDouble())
fun Klock.currentTimeMicro() = millisecondsToMicroseconds(currentTimeMillis())
fun Klock.millisecondsToMicroseconds(millis: Long) = millis * 1000L
fun Klock.millisecondsToMicroseconds(millis: Double) = millis * 1000.0