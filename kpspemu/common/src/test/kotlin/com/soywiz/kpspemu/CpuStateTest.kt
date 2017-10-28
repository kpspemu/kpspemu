package com.soywiz.kpspemu

import org.junit.Test
import kotlin.test.assertEquals

class CpuStateTest {
	val s = CpuState()

	@Test
	fun gpr0IsAlways0() {
		s.GPR[0] = 1
		s.r0 = 1
		assertEquals(0, s.GPR[0])
		assertEquals(0, s.r0)
	}
}