package com.soywiz.kpspemu.hle.manager

import com.soywiz.klock.Klock
import com.soywiz.kpspemu.Emulator
import com.soywiz.kpspemu.WithEmulator

class TimeManager(override val emulator: Emulator) : WithEmulator {
	fun getTimeInMicroseconds(): Long = Klock.currentTimeMillis()
	fun getTimeInMicrosecondsDouble(): Double = Klock.currentTimeMillisDouble()
	fun getTimeInSeconds(): Int = (Klock.currentTimeMillisDouble() / 1000).toInt()
}