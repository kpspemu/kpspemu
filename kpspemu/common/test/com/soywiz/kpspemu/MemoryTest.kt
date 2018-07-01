package com.soywiz.kpspemu

import com.soywiz.kpspemu.mem.*
import kotlin.test.*

class MemoryTest {
    @Test
    fun name() {
        val mem = Memory()
        mem.lb(1)
    }
}