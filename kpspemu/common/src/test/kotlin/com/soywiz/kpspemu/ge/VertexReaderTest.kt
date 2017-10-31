package com.soywiz.kpspemu.ge

import com.soywiz.korio.crypto.Hex
import com.soywiz.korio.stream.openSync
import org.junit.Test
import kotlin.test.assertEquals

class VertexReaderTest {
	@Test
	fun testCubeSampleVertices() {
		val vertices = Hex.decode("000000000000000000007fff000080bf000080bf0000803f0000803f0000000000007fff000080bf0000803f0000803f0000803f0000803f00007fff0000803f0000803f0000803f000000000000000000007fff000080bf000080bf0000803f0000803f0000803f00007fff0000803f0000803f0000803f000000000000803f00007fff0000803f000080bf0000803f000000000000000000007fff000080bf000080bf000080bf0000803f0000000000007fff0000803f000080bf000080bf0000803f0000803f00007fff0000803f0000803f000080bf000000000000000000007fff000080bf000080bf000080bf0000803f0000803f00007fff0000803f0000803f000080bf000000000000803f00007fff000080bf0000803f000080bf0000000000000000007f00ff0000803f000080bf000080bf0000803f00000000007f00ff0000803f000080bf0000803f0000803f0000803f007f00ff0000803f0000803f0000803f0000000000000000007f00ff0000803f000080bf000080bf0000803f0000803f007f00ff0000803f0000803f0000803f000000000000803f007f00ff0000803f0000803f000080bf0000000000000000007f00ff000080bf000080bf000080bf0000803f00000000007f00ff000080bf0000803f000080bf0000803f0000803f007f00ff000080bf0000803f0000803f0000000000000000007f00ff000080bf000080bf000080bf0000803f0000803f007f00ff000080bf0000803f0000803f000000000000803f007f00ff000080bf000080bf0000803f00000000000000007f0000ff000080bf0000803f000080bf0000803f000000007f0000ff0000803f0000803f000080bf0000803f0000803f7f0000ff0000803f0000803f0000803f00000000000000007f0000ff000080bf0000803f000080bf0000803f0000803f7f0000ff0000803f0000803f0000803f000000000000803f7f0000ff000080bf0000803f0000803f00000000000000007f0000ff000080bf000080bf000080bf0000803f000000007f0000ff000080bf000080bf0000803f0000803f0000803f7f0000ff0000803f000080bf0000803f00000000000000007f0000ff000080bf000080bf000080bf0000803f0000803f7f0000ff0000803f000080bf0000803f000000000000803f7f0000ff0000803f000080bf000080bf")
		val s = vertices.openSync()
		val vtype = VertexType(0x0000019F)
		val reader = VertexReader()
		fun readOne() = reader.readOne(s, vtype).toString()
		val numberOfVertices = s.length / vtype.size()

		//println(numberOfVertices)
		//for (n in 0 until numberOfVertices) println(readOne())

		assertEquals(36, numberOfVertices)
		assertEquals("VertexRaw(0xFF7F0000, normal=[0.0, 0.0, 0.0], pos=[-1.0, -1.0, 1.0], tex=[0.0, 0.0, 0.0], weights=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])", readOne())
		assertEquals("VertexRaw(0xFF7F0000, normal=[0.0, 0.0, 0.0], pos=[-1.0, 1.0, 1.0], tex=[1.0, 0.0, 0.0], weights=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])", readOne())
		assertEquals("VertexRaw(0xFF7F0000, normal=[0.0, 0.0, 0.0], pos=[1.0, 1.0, 1.0], tex=[1.0, 1.0, 0.0], weights=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])", readOne())
		assertEquals("VertexRaw(0xFF7F0000, normal=[0.0, 0.0, 0.0], pos=[-1.0, -1.0, 1.0], tex=[0.0, 0.0, 0.0], weights=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])", readOne())
		assertEquals("VertexRaw(0xFF7F0000, normal=[0.0, 0.0, 0.0], pos=[1.0, 1.0, 1.0], tex=[1.0, 1.0, 0.0], weights=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])", readOne())
		assertEquals("VertexRaw(0xFF7F0000, normal=[0.0, 0.0, 0.0], pos=[1.0, -1.0, 1.0], tex=[0.0, 1.0, 0.0], weights=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])", readOne())
		assertEquals("VertexRaw(0xFF7F0000, normal=[0.0, 0.0, 0.0], pos=[-1.0, -1.0, -1.0], tex=[0.0, 0.0, 0.0], weights=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])", readOne())
		assertEquals("VertexRaw(0xFF7F0000, normal=[0.0, 0.0, 0.0], pos=[1.0, -1.0, -1.0], tex=[1.0, 0.0, 0.0], weights=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])", readOne())
		assertEquals("VertexRaw(0xFF7F0000, normal=[0.0, 0.0, 0.0], pos=[1.0, 1.0, -1.0], tex=[1.0, 1.0, 0.0], weights=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])", readOne())
		assertEquals("VertexRaw(0xFF7F0000, normal=[0.0, 0.0, 0.0], pos=[-1.0, -1.0, -1.0], tex=[0.0, 0.0, 0.0], weights=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])", readOne())
		assertEquals("VertexRaw(0xFF7F0000, normal=[0.0, 0.0, 0.0], pos=[1.0, 1.0, -1.0], tex=[1.0, 1.0, 0.0], weights=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])", readOne())
		assertEquals("VertexRaw(0xFF7F0000, normal=[0.0, 0.0, 0.0], pos=[-1.0, 1.0, -1.0], tex=[0.0, 1.0, 0.0], weights=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])", readOne())
		assertEquals("VertexRaw(0xFF007F00, normal=[0.0, 0.0, 0.0], pos=[1.0, -1.0, -1.0], tex=[0.0, 0.0, 0.0], weights=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])", readOne())
		assertEquals("VertexRaw(0xFF007F00, normal=[0.0, 0.0, 0.0], pos=[1.0, -1.0, 1.0], tex=[1.0, 0.0, 0.0], weights=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])", readOne())
		assertEquals("VertexRaw(0xFF007F00, normal=[0.0, 0.0, 0.0], pos=[1.0, 1.0, 1.0], tex=[1.0, 1.0, 0.0], weights=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])", readOne())
		assertEquals("VertexRaw(0xFF007F00, normal=[0.0, 0.0, 0.0], pos=[1.0, -1.0, -1.0], tex=[0.0, 0.0, 0.0], weights=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])", readOne())
		assertEquals("VertexRaw(0xFF007F00, normal=[0.0, 0.0, 0.0], pos=[1.0, 1.0, 1.0], tex=[1.0, 1.0, 0.0], weights=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])", readOne())
		assertEquals("VertexRaw(0xFF007F00, normal=[0.0, 0.0, 0.0], pos=[1.0, 1.0, -1.0], tex=[0.0, 1.0, 0.0], weights=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])", readOne())
		assertEquals("VertexRaw(0xFF007F00, normal=[0.0, 0.0, 0.0], pos=[-1.0, -1.0, -1.0], tex=[0.0, 0.0, 0.0], weights=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])", readOne())
		assertEquals("VertexRaw(0xFF007F00, normal=[0.0, 0.0, 0.0], pos=[-1.0, 1.0, -1.0], tex=[1.0, 0.0, 0.0], weights=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])", readOne())
		assertEquals("VertexRaw(0xFF007F00, normal=[0.0, 0.0, 0.0], pos=[-1.0, 1.0, 1.0], tex=[1.0, 1.0, 0.0], weights=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])", readOne())
		assertEquals("VertexRaw(0xFF007F00, normal=[0.0, 0.0, 0.0], pos=[-1.0, -1.0, -1.0], tex=[0.0, 0.0, 0.0], weights=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])", readOne())
		assertEquals("VertexRaw(0xFF007F00, normal=[0.0, 0.0, 0.0], pos=[-1.0, 1.0, 1.0], tex=[1.0, 1.0, 0.0], weights=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])", readOne())
		assertEquals("VertexRaw(0xFF007F00, normal=[0.0, 0.0, 0.0], pos=[-1.0, -1.0, 1.0], tex=[0.0, 1.0, 0.0], weights=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])", readOne())
		assertEquals("VertexRaw(0xFF00007F, normal=[0.0, 0.0, 0.0], pos=[-1.0, 1.0, -1.0], tex=[0.0, 0.0, 0.0], weights=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])", readOne())
		assertEquals("VertexRaw(0xFF00007F, normal=[0.0, 0.0, 0.0], pos=[1.0, 1.0, -1.0], tex=[1.0, 0.0, 0.0], weights=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])", readOne())
		assertEquals("VertexRaw(0xFF00007F, normal=[0.0, 0.0, 0.0], pos=[1.0, 1.0, 1.0], tex=[1.0, 1.0, 0.0], weights=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])", readOne())
		assertEquals("VertexRaw(0xFF00007F, normal=[0.0, 0.0, 0.0], pos=[-1.0, 1.0, -1.0], tex=[0.0, 0.0, 0.0], weights=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])", readOne())
		assertEquals("VertexRaw(0xFF00007F, normal=[0.0, 0.0, 0.0], pos=[1.0, 1.0, 1.0], tex=[1.0, 1.0, 0.0], weights=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])", readOne())
		assertEquals("VertexRaw(0xFF00007F, normal=[0.0, 0.0, 0.0], pos=[-1.0, 1.0, 1.0], tex=[0.0, 1.0, 0.0], weights=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])", readOne())
		assertEquals("VertexRaw(0xFF00007F, normal=[0.0, 0.0, 0.0], pos=[-1.0, -1.0, -1.0], tex=[0.0, 0.0, 0.0], weights=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])", readOne())
		assertEquals("VertexRaw(0xFF00007F, normal=[0.0, 0.0, 0.0], pos=[-1.0, -1.0, 1.0], tex=[1.0, 0.0, 0.0], weights=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])", readOne())
		assertEquals("VertexRaw(0xFF00007F, normal=[0.0, 0.0, 0.0], pos=[1.0, -1.0, 1.0], tex=[1.0, 1.0, 0.0], weights=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])", readOne())
		assertEquals("VertexRaw(0xFF00007F, normal=[0.0, 0.0, 0.0], pos=[-1.0, -1.0, -1.0], tex=[0.0, 0.0, 0.0], weights=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])", readOne())
		assertEquals("VertexRaw(0xFF00007F, normal=[0.0, 0.0, 0.0], pos=[1.0, -1.0, 1.0], tex=[1.0, 1.0, 0.0], weights=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])", readOne())
		assertEquals("VertexRaw(0xFF00007F, normal=[0.0, 0.0, 0.0], pos=[1.0, -1.0, -1.0], tex=[0.0, 1.0, 0.0], weights=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])", readOne())

	}

	@Test
	fun testClearFastVerticesSpriteRaw() {
		val vertices = Hex.decode("000000000000000000003f5b33445500e0011001000000d2")
		val s = vertices.openSync()
		val vtype = VertexType(0x0080011C)
		val reader = VertexReader()
		fun readOne() = reader.readOne(s, vtype).toString()
		val numberOfVertices = s.length / vtype.size()

		//println(numberOfVertices)
		//for (n in 0 until numberOfVertices) println(readOne())

		assertEquals(2, numberOfVertices)
		assertEquals("VertexRaw(0x00000000, normal=[0.0, 0.0, 0.0], pos=[0.0, 0.0, 0.0], tex=[0.0, 0.0, 0.0], weights=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])", readOne())
		assertEquals("VertexRaw(0x00554433, normal=[0.0, 0.0, 0.0], pos=[480.0, 272.0, 0.0], tex=[0.0, 0.0, 0.0], weights=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])", readOne())
	}

	@Test
	fun testClearFastVerticesAfterExpansion() {
		val vertices = Hex.decode("000000000000000000003f5b33445500e0011001000000d23344550000001001000000d233445500e0010000000000d2")
		val s = vertices.openSync()
		val vtype = VertexType(0x0080011C)
		val reader = VertexReader()
		fun readOne() = reader.readOne(s, vtype).toString()
		val numberOfVertices = s.length / vtype.size()

		println(numberOfVertices)
		for (n in 0 until numberOfVertices) println(readOne())
	}
}