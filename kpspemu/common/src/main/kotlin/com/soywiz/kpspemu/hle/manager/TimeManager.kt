package com.soywiz.kpspemu.hle.manager

import com.soywiz.klock.Klock
import com.soywiz.kpspemu.Emulator
import com.soywiz.kpspemu.WithEmulator
import com.soywiz.kpspemu.util.millisecondsToMicroseconds

class TimeManager(override val emulator: Emulator) : WithEmulator {
	fun getTimeInMicroseconds(): Long = Klock.millisecondsToMicroseconds(Klock.currentTimeMillis())
}