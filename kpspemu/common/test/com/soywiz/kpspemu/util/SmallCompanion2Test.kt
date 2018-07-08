package com.soywiz.kpspemu.util

import com.soywiz.korio.util.*
import com.soywiz.kpspemu.*
import kotlin.test.*

class SmallCompanion2Test : BaseTest() {
    enum class WaveformEffectType(override val id: Int) : IdEnum {
        OFF(-1), ROOM(0), UNK1(1), UNK2(2), UNK3(3),
        HALL(4), SPACE(5), ECHO(6), DELAY(7), PIPE(8);

        companion object : SmallCompanion2<WaveformEffectType>(values())
    }

    @Test
    fun name() {
        assertEquals(WaveformEffectType.OFF, WaveformEffectType(-1))
        assertEquals(WaveformEffectType.ROOM, WaveformEffectType(0))
        assertEquals(WaveformEffectType.PIPE, WaveformEffectType(8))
        assertEquals(WaveformEffectType.OFF, WaveformEffectType(-2))
        assertEquals(WaveformEffectType.OFF, WaveformEffectType(9))
    }
}