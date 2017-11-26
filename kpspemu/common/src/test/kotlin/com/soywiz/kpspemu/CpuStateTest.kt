package com.soywiz.kpspemu

import com.soywiz.kpspemu.cpu.CpuState
import com.soywiz.kpspemu.cpu.GlobalCpuState
import com.soywiz.kpspemu.mem.DummyMemory
import org.junit.Test
import kotlin.test.assertEquals

class CpuStateTest {
	val s = CpuState(GlobalCpuState(), DummyMemory)

	@Test
	fun gpr0IsAlways0() {
		s.setGpr(0, 1)
		//s.r0 = 1
		assertEquals(0, s.getGpr(0))
		//mytest.assertEquals(0, s.r0)
	}

	@Test
	fun otherGprCanSetToOtherValues() {
		for (n in 1 until 32) s.setGpr(n, n * 1000)
		for (n in 1 until 32) assertEquals(n * 1000, s.getGpr(n))
	}
}