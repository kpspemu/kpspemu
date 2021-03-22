package com.soywiz.kpspemu.format

import com.soywiz.korio.file.std.*
import com.soywiz.korio.stream.*
import com.soywiz.kpspemu.*
import com.soywiz.kpspemu.format.elf.*
import kotlin.test.*

class PbpTest : BaseTest() {
    @Test
    fun name() = pspSuspendTest {
        val pbp = Pbp.load(resourcesVfs["lines.pbp"].open())
        assertEquals(listOf(408L, 0L, 0L, 0L, 0L, 0L, 30280L, 0L), pbp.streams.map { it.size() })
        assertEquals(408, pbp[Pbp.PARAM_SFO]!!.readAll().size)
        assertEquals(408, pbp[Pbp.PARAM_SFO]!!.readAll().size, "read twice")
        assertEquals(30280, pbp[Pbp.PSP_DATA]!!.readAll().size)
        val elf = Elf.fromStream(pbp[Pbp.PSP_DATA]!!.readAll().openSync())
    }
}