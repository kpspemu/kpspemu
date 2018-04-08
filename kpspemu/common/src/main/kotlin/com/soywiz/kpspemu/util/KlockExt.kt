package com.soywiz.kpspemu.util

import com.soywiz.klock.*

fun Klock.currentTimeMicroInt(): Int {
    val millisDouble = currentTimeMillisDouble()
    //val millisInt = millisDouble.toInt()
    val millisInt = (millisDouble % 0x7FFFFFFF.toDouble()).toInt()
    val res = millisecondsToMicroseconds(millisInt)
    //println("$millisDouble, $millisInt, $res")
    return res
}

fun Klock.currentTimeMicroDouble() = millisecondsToMicroseconds(currentTimeMillisDouble())
fun Klock.currentTimeMicro() = millisecondsToMicroseconds(currentTimeMillis())
fun Klock.millisecondsToMicroseconds(millis: Long) = millis * 1000L
fun Klock.millisecondsToMicroseconds(millis: Double) = millis * 1000.0
fun Klock.millisecondsToMicroseconds(millis: Int) = millis * 1000
