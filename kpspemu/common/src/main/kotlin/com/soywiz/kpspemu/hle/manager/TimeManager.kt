package com.soywiz.kpspemu.hle.manager

import com.soywiz.klock.DateTime
import com.soywiz.klock.Klock
import com.soywiz.korio.stream.*
import com.soywiz.kpspemu.Emulator
import com.soywiz.kpspemu.WithEmulator
import com.soywiz.kpspemu.util.currentTimeMicro
import com.soywiz.kpspemu.util.currentTimeMicroDouble
import com.soywiz.kpspemu.util.currentTimeMicroInt

class TimeManager(override val emulator: Emulator) : WithEmulator {
	fun getTimeInMillisecondsDouble(): Double = Klock.currentTimeMillisDouble()
	fun getTimeInMilliseconds(): Long = Klock.currentTimeMillis()
	fun getTimeInMicroseconds(): Long = Klock.currentTimeMicro()
	fun getTimeInMicrosecondsDouble(): Double = Klock.currentTimeMicroDouble()
	fun getTimeInMicrosecondsInt(): Int = Klock.currentTimeMicroInt()
	fun getTimeInSeconds(): Int = (Klock.currentTimeMillisDouble() / 1000).toInt()
}

class ScePspDateTime(
	var year: Int,
	var month: Int,
	var day: Int,
	var hour: Int,
	var minute: Int,
	var second: Int,
	var microsecond: Int
) {
	val date: DateTime get() = DateTime.createAdjusted(year, month, day, hour, minute, second, microsecond / 1000)

	constructor(date: DateTime) : this(date.year, date.month, date.dayOfMonth, date.hours, date.minutes, date.seconds, date.milliseconds * 1000)
	constructor(ticks: Long) : this(DateTime(ticks))

	companion object {
		fun read(s: SyncStream): ScePspDateTime = s.run {
			ScePspDateTime(
				year = s.readU16_le(),
				month = s.readU16_le(),
				day = s.readU16_le(),
				hour = s.readU16_le(),
				minute = s.readU16_le(),
				second = s.readU16_le(),
				microsecond = s.readS32_le()
			)
		}
	}

	fun write(s: SyncStream) = s.apply {
		write16_le(year)
		write16_le(month)
		write16_le(day)
		write16_le(hour)
		write16_le(minute)
		write16_le(second)
		write32_le(microsecond)
	}
}
