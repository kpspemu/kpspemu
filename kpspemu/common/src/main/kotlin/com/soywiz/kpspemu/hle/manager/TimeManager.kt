package com.soywiz.kpspemu.hle.manager

import com.soywiz.klock.Klock
import com.soywiz.kpspemu.Emulator
import com.soywiz.kpspemu.WithEmulator
import com.soywiz.kpspemu.util.currentTimeMicro
import com.soywiz.kpspemu.util.currentTimeMicroDouble
import com.soywiz.kpspemu.util.currentTimeMicroInt

class TimeManager(override val emulator: Emulator) : WithEmulator {
	fun getTimeInMicroseconds(): Long = Klock.currentTimeMicro()
	fun getTimeInMicrosecondsDouble(): Double = Klock.currentTimeMicroDouble()
	fun getTimeInMicrosecondsInt(): Int = Klock.currentTimeMicroInt()
	fun getTimeInSeconds(): Int = (Klock.currentTimeMillisDouble() / 1000).toInt()
}