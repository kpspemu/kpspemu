package com.soywiz.kpspemu.hle.manager

import com.soywiz.klock.DateTime
import com.soywiz.klock.Klock
import com.soywiz.korio.stream.SyncStream
import com.soywiz.kpspemu.Emulator
import com.soywiz.kpspemu.WithEmulator
import com.soywiz.kpspemu.util.*

class TimeManager(override val emulator: Emulator) : WithEmulator {
	fun getTimeInMillisecondsDouble(): Double = Klock.currentTimeMillisDouble()
	fun getTimeInMilliseconds(): Long = Klock.currentTimeMillis()
	fun getTimeInMicroseconds(): Long = Klock.currentTimeMicro()
	fun getTimeInMicrosecondsDouble(): Double = Klock.currentTimeMicroDouble()
	fun getTimeInMicrosecondsInt(): Int = Klock.currentTimeMicroInt()
	fun getTimeInSeconds(): Int = (Klock.currentTimeMillisDouble() / 1000.0).toInt()
	fun reset() {
	}
}

data class ScePspDateTime(
	var year: Int = 0,
	var month: Int = 0,
	var day: Int = 0,
	var hour: Int = 0,
	var minute: Int = 0,
	var second: Int = 0,
	var microsecond: Int = 0
) {
	val date: DateTime get() = DateTime.createAdjusted(year, month, day, hour, minute, second, microsecond / 1000)

	constructor(date: DateTime) : this(date.year, date.month, date.dayOfMonth, date.hours, date.minutes, date.seconds, date.milliseconds * 1000)
	constructor(ticks: Long) : this(DateTime(ticks))

	companion object : Struct<ScePspDateTime>({ ScePspDateTime(0L) },
		ScePspDateTime::year AS UINT16,
		ScePspDateTime::month AS UINT16,
		ScePspDateTime::day AS UINT16,
		ScePspDateTime::hour AS UINT16,
		ScePspDateTime::minute AS UINT16,
		ScePspDateTime::second AS UINT16,
		ScePspDateTime::microsecond AS INT32
	)

	fun write(s: SyncStream) = s.write(ScePspDateTime, this)
}
