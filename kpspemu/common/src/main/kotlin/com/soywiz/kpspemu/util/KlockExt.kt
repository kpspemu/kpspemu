package com.soywiz.kpspemu.util

import com.soywiz.klock.Klock

fun Klock.currentTimeMicro() = millisecondsToMicroseconds(currentTimeMillis())
fun Klock.millisecondsToMicroseconds(millis: Long) = millis * 1000L