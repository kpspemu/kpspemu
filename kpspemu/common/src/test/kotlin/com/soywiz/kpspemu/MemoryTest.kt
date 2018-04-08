package com.soywiz.kpspemu

import com.soywiz.kpspemu.mem.*
import org.junit.*

class MemoryTest {
    @Test
    fun name() {
        val mem = Memory()
        mem.lb(1)
    }
}