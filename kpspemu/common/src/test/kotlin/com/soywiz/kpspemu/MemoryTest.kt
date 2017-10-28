package com.soywiz.kpspemu

import com.soywiz.kpspemu.mem.Memory
import org.junit.Test

class MemoryTest {
	@Test
	fun name() {
		val mem = Memory()
		mem.lb(1)
	}
}