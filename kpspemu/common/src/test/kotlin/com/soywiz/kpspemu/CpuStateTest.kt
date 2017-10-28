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

	@Test
	fun otherGprCanSetToOtherValues() {
		for (n in 1 until 32) s.GPR[n] = n * 1000
		for (n in 1 until 32) assertEquals(n * 1000, s.GPR[n])
	}
}