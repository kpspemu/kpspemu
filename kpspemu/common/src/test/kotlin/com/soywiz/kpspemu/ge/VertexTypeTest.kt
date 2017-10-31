package com.soywiz.kpspemu.ge

import org.junit.Test
import kotlin.test.assertEquals

class VertexTypeTest {
	@Test
	fun testCubeExampleVertexType() {
		assertEquals(
			"VertexType(color=COLOR8888, normal=VOID, pos=FLOAT, tex=FLOAT, weight=VOID, size=24)",
			VertexType(0x0000019F).toString()
		)
	}

	@Test
	fun testClearVertexType() {
		assertEquals(
			"VertexType(color=COLOR8888, normal=VOID, pos=SHORT, tex=VOID, weight=VOID, size=8)",
			VertexType(0x0080011C).toString()
		)
	}
}