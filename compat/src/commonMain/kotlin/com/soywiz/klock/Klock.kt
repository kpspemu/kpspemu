package com.soywiz.klock

object Klock {
    fun currentTimeMillisDouble(): Double {
        return DateTime.nowUnix()
    }
    fun currentTimeMillis(): Long {
        return DateTime.nowUnix().toLong()
    }
}

typealias TimeDistance = DateTimeSpan
