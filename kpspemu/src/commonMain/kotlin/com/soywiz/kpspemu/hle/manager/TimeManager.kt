package com.soywiz.kpspemu.hle.manager

import com.soywiz.klock.*
import com.soywiz.korio.stream.*
import com.soywiz.kpspemu.*
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

//val Number.microseconds get() = (this.toDouble() / 1000000.0).seconds

data class ScePspDateTime(
    var year: Int = 0,
    var month: Int = 0,
    var day: Int = 0,
    var hour: Int = 0,
    var minute: Int = 0,
    var second: Int = 0,
    var microsecond: Int = 0
) {
    //val date: DateTime get() = DateTime.createAdjusted(year, month, day, hour, minute, second, microsecond / 1000)
    val date: DateTime get() = DateTime.createClamped(year, month, day, hour, minute, second, microsecond / 1000)
    val microAdjust: Int get() = (microsecond % 1000)
    val tick: Long get() = EPOCH_TICKS + (date.unixMillisLong * 1000L) + microAdjust

    companion object : Struct<ScePspDateTime>(
        { ScePspDateTime(0L) },
        ScePspDateTime::year AS UINT16,
        ScePspDateTime::month AS UINT16,
        ScePspDateTime::day AS UINT16,
        ScePspDateTime::hour AS UINT16,
        ScePspDateTime::minute AS UINT16,
        ScePspDateTime::second AS UINT16,
        ScePspDateTime::microsecond AS INT32
    ) {
        val EPOCH_TICKS = 62135596800000000L
        operator fun invoke(date: DateTime, microAdjust: Int = 0): ScePspDateTime {
            return ScePspDateTime(date.local, microAdjust)
        }
        operator fun invoke(date: DateTimeTz, microAdjust: Int = 0): ScePspDateTime {
            return ScePspDateTime(
                date.yearInt,
                date.month1,
                date.dayOfMonth,
                date.hours,
                date.minutes,
                date.seconds,
                date.milliseconds * 1000 + microAdjust
            )
        }

        operator fun invoke(ticks: Long): ScePspDateTime {
            val epochAdjust = ticks - EPOCH_TICKS
            return ScePspDateTime(DateTime.fromUnix(epochAdjust / 1000).local, microAdjust = (epochAdjust % 1000).toInt())
        }
    }

    fun write(s: SyncStream) = s.write(ScePspDateTime, this)
}
