package com.soywiz.kpspemu.ge

import com.soywiz.korio.stream.*
import com.soywiz.kpspemu.*
import com.soywiz.krypto.encoding.*
import kotlin.test.*

class VertexReaderTest : BaseTest() {
    @Test
    fun testCubeSampleVertices() {
        val vertices =
            Hex.decode("000000000000000000007fff000080bf000080bf0000803f0000803f0000000000007fff000080bf0000803f0000803f0000803f0000803f00007fff0000803f0000803f0000803f000000000000000000007fff000080bf000080bf0000803f0000803f0000803f00007fff0000803f0000803f0000803f000000000000803f00007fff0000803f000080bf0000803f000000000000000000007fff000080bf000080bf000080bf0000803f0000000000007fff0000803f000080bf000080bf0000803f0000803f00007fff0000803f0000803f000080bf000000000000000000007fff000080bf000080bf000080bf0000803f0000803f00007fff0000803f0000803f000080bf000000000000803f00007fff000080bf0000803f000080bf0000000000000000007f00ff0000803f000080bf000080bf0000803f00000000007f00ff0000803f000080bf0000803f0000803f0000803f007f00ff0000803f0000803f0000803f0000000000000000007f00ff0000803f000080bf000080bf0000803f0000803f007f00ff0000803f0000803f0000803f000000000000803f007f00ff0000803f0000803f000080bf0000000000000000007f00ff000080bf000080bf000080bf0000803f00000000007f00ff000080bf0000803f000080bf0000803f0000803f007f00ff000080bf0000803f0000803f0000000000000000007f00ff000080bf000080bf000080bf0000803f0000803f007f00ff000080bf0000803f0000803f000000000000803f007f00ff000080bf000080bf0000803f00000000000000007f0000ff000080bf0000803f000080bf0000803f000000007f0000ff0000803f0000803f000080bf0000803f0000803f7f0000ff0000803f0000803f0000803f00000000000000007f0000ff000080bf0000803f000080bf0000803f0000803f7f0000ff0000803f0000803f0000803f000000000000803f7f0000ff000080bf0000803f0000803f00000000000000007f0000ff000080bf000080bf000080bf0000803f000000007f0000ff000080bf000080bf0000803f0000803f0000803f7f0000ff0000803f000080bf0000803f00000000000000007f0000ff000080bf000080bf000080bf0000803f0000803f7f0000ff0000803f000080bf0000803f000000000000803f7f0000ff0000803f000080bf000080bf")
        val s = vertices.openSync()
        val vtype = VertexType(0x0000019F)
        val reader = VertexReader()
        fun readOne() = reader.readOne(s, vtype).toString()
        val numberOfVertices = s.length / vtype.size

        //println(numberOfVertices)
        //for (n in 0 until numberOfVertices) println(readOne())

        assertEquals(36, numberOfVertices)
        assertEquals("VertexRaw(color=0xFF7F0000, pos=[-1.0, -1.0, 1.0], tex=[0.0, 0.0, 0.0])", readOne())
        assertEquals("VertexRaw(color=0xFF7F0000, pos=[-1.0, 1.0, 1.0], tex=[1.0, 0.0, 0.0])", readOne())
        assertEquals("VertexRaw(color=0xFF7F0000, pos=[1.0, 1.0, 1.0], tex=[1.0, 1.0, 0.0])", readOne())
        assertEquals("VertexRaw(color=0xFF7F0000, pos=[-1.0, -1.0, 1.0], tex=[0.0, 0.0, 0.0])", readOne())
        assertEquals("VertexRaw(color=0xFF7F0000, pos=[1.0, 1.0, 1.0], tex=[1.0, 1.0, 0.0])", readOne())
        assertEquals("VertexRaw(color=0xFF7F0000, pos=[1.0, -1.0, 1.0], tex=[0.0, 1.0, 0.0])", readOne())
        assertEquals("VertexRaw(color=0xFF7F0000, pos=[-1.0, -1.0, -1.0], tex=[0.0, 0.0, 0.0])", readOne())
        assertEquals("VertexRaw(color=0xFF7F0000, pos=[1.0, -1.0, -1.0], tex=[1.0, 0.0, 0.0])", readOne())
        assertEquals("VertexRaw(color=0xFF7F0000, pos=[1.0, 1.0, -1.0], tex=[1.0, 1.0, 0.0])", readOne())
        assertEquals("VertexRaw(color=0xFF7F0000, pos=[-1.0, -1.0, -1.0], tex=[0.0, 0.0, 0.0])", readOne())
        assertEquals("VertexRaw(color=0xFF7F0000, pos=[1.0, 1.0, -1.0], tex=[1.0, 1.0, 0.0])", readOne())
        assertEquals("VertexRaw(color=0xFF7F0000, pos=[-1.0, 1.0, -1.0], tex=[0.0, 1.0, 0.0])", readOne())
        assertEquals("VertexRaw(color=0xFF007F00, pos=[1.0, -1.0, -1.0], tex=[0.0, 0.0, 0.0])", readOne())
        assertEquals("VertexRaw(color=0xFF007F00, pos=[1.0, -1.0, 1.0], tex=[1.0, 0.0, 0.0])", readOne())
        assertEquals("VertexRaw(color=0xFF007F00, pos=[1.0, 1.0, 1.0], tex=[1.0, 1.0, 0.0])", readOne())
        assertEquals("VertexRaw(color=0xFF007F00, pos=[1.0, -1.0, -1.0], tex=[0.0, 0.0, 0.0])", readOne())
        assertEquals("VertexRaw(color=0xFF007F00, pos=[1.0, 1.0, 1.0], tex=[1.0, 1.0, 0.0])", readOne())
        assertEquals("VertexRaw(color=0xFF007F00, pos=[1.0, 1.0, -1.0], tex=[0.0, 1.0, 0.0])", readOne())
        assertEquals("VertexRaw(color=0xFF007F00, pos=[-1.0, -1.0, -1.0], tex=[0.0, 0.0, 0.0])", readOne())
        assertEquals("VertexRaw(color=0xFF007F00, pos=[-1.0, 1.0, -1.0], tex=[1.0, 0.0, 0.0])", readOne())
        assertEquals("VertexRaw(color=0xFF007F00, pos=[-1.0, 1.0, 1.0], tex=[1.0, 1.0, 0.0])", readOne())
        assertEquals("VertexRaw(color=0xFF007F00, pos=[-1.0, -1.0, -1.0], tex=[0.0, 0.0, 0.0])", readOne())
        assertEquals("VertexRaw(color=0xFF007F00, pos=[-1.0, 1.0, 1.0], tex=[1.0, 1.0, 0.0])", readOne())
        assertEquals("VertexRaw(color=0xFF007F00, pos=[-1.0, -1.0, 1.0], tex=[0.0, 1.0, 0.0])", readOne())
        assertEquals("VertexRaw(color=0xFF00007F, pos=[-1.0, 1.0, -1.0], tex=[0.0, 0.0, 0.0])", readOne())
        assertEquals("VertexRaw(color=0xFF00007F, pos=[1.0, 1.0, -1.0], tex=[1.0, 0.0, 0.0])", readOne())
        assertEquals("VertexRaw(color=0xFF00007F, pos=[1.0, 1.0, 1.0], tex=[1.0, 1.0, 0.0])", readOne())
        assertEquals("VertexRaw(color=0xFF00007F, pos=[-1.0, 1.0, -1.0], tex=[0.0, 0.0, 0.0])", readOne())
        assertEquals("VertexRaw(color=0xFF00007F, pos=[1.0, 1.0, 1.0], tex=[1.0, 1.0, 0.0])", readOne())
        assertEquals("VertexRaw(color=0xFF00007F, pos=[-1.0, 1.0, 1.0], tex=[0.0, 1.0, 0.0])", readOne())
        assertEquals("VertexRaw(color=0xFF00007F, pos=[-1.0, -1.0, -1.0], tex=[0.0, 0.0, 0.0])", readOne())
        assertEquals("VertexRaw(color=0xFF00007F, pos=[-1.0, -1.0, 1.0], tex=[1.0, 0.0, 0.0])", readOne())
        assertEquals("VertexRaw(color=0xFF00007F, pos=[1.0, -1.0, 1.0], tex=[1.0, 1.0, 0.0])", readOne())
        assertEquals("VertexRaw(color=0xFF00007F, pos=[-1.0, -1.0, -1.0], tex=[0.0, 0.0, 0.0])", readOne())
        assertEquals("VertexRaw(color=0xFF00007F, pos=[1.0, -1.0, 1.0], tex=[1.0, 1.0, 0.0])", readOne())
        assertEquals("VertexRaw(color=0xFF00007F, pos=[1.0, -1.0, -1.0], tex=[0.0, 1.0, 0.0])", readOne())

    }

    @Test
    fun testClearFastVerticesSpriteRaw() {
        val vertices = Hex.decode("000000000000000000003f5b33445500e0011001000000d2")
        val s = vertices.openSync()
        val vtype = VertexType(0x0080011C)
        val reader = VertexReader()
        fun readOne() = reader.readOne(s, vtype).toString()
        val numberOfVertices = s.length / vtype.size

        //println(numberOfVertices)
        //for (n in 0 until numberOfVertices) println(readOne())

        assertEquals(2, numberOfVertices)
        assertEquals("VertexRaw(color=0x00000000, pos=[0.0, 0.0, 0.0])", readOne())
        assertEquals("VertexRaw(color=0x00554433, pos=[480.0, 272.0, 0.0])", readOne())
    }

    @Test
    fun testClearFastVerticesAfterExpansion() {
        val vertices =
            Hex.decode("334455000000000000003f5b33445500e0011001000000d23344550000001001000000d233445500e0010000000000d2")
        val s = vertices.openSync()
        val vtype = VertexType(0x0080011C)
        val reader = VertexReader()
        fun readOne() = reader.readOne(s, vtype).toString()
        val numberOfVertices = s.length / vtype.size

        //println(numberOfVertices)
        //for (n in 0 until numberOfVertices) println(readOne())

        assertEquals(4, numberOfVertices)
        assertEquals("VertexRaw(color=0x00554433, pos=[0.0, 0.0, 0.0])", readOne())
        assertEquals("VertexRaw(color=0x00554433, pos=[480.0, 272.0, 0.0])", readOne())
        assertEquals("VertexRaw(color=0x00554433, pos=[0.0, 272.0, 0.0])", readOne())
        assertEquals("VertexRaw(color=0x00554433, pos=[480.0, 0.0, 0.0])", readOne())
    }

    @Test
    fun testCavestoryBgSprite() {
        //[AGRenderer]: indices: [0, 3, 2, 2, 3, 1]
        //[AGRenderer]: primitive: TRIANGLES
        //[AGRenderer]: vertexCount: 6
        //[AGRenderer]: vertexType: 0x00800102
        //[AGRenderer]: vertices: 0000000000000000000000000000e001100100000000000000001001000000000000e00100000000
        //[AGRenderer]: matrix: Matrix4([0.004166667, 0.0, 0.0, 0.0, 0.0, -0.007352941, 0.0, 0.0, 0.0, 0.0, 3.0518044E-5, 0.0, -1.0, 1.0, -1.0, 1.0])
        val vertices = Hex.decode("0000000000000000000000000000e001100100000000000000001001000000000000e00100000000")
        val s = vertices.openSync()
        val vtype = VertexType(0x00800102)
        val reader = VertexReader()
        fun readOne() = reader.readOne(s, vtype).toString()
        val numberOfVertices = s.length / vtype.size

        //println(numberOfVertices)
        //println(s.length)
        //println(vtype.size)
        //println(vtype)
        //for (n in 0 until numberOfVertices) println(readOne())

        //assertEquals(4, numberOfVertices)
        //assertEquals("VertexRaw(0x00554433, normal=[0.0, 0.0, 0.0], pos=[0.0, 0.0, 0.0], tex=[0.0, 0.0, 0.0])", readOne())
        //assertEquals("VertexRaw(0x00554433, normal=[0.0, 0.0, 0.0], pos=[480.0, 272.0, 0.0], tex=[0.0, 0.0, 0.0])", readOne())
        //assertEquals("VertexRaw(0x00554433, normal=[0.0, 0.0, 0.0], pos=[0.0, 272.0, 0.0], tex=[0.0, 0.0, 0.0])", readOne())
        //assertEquals("VertexRaw(0x00554433, normal=[0.0, 0.0, 0.0], pos=[480.0, 0.0, 0.0], tex=[0.0, 0.0, 0.0])", readOne())

        assertEquals("VertexRaw(pos=[0.0, 0.0, 0.0], tex=[0.0, 0.0, 0.0])", readOne())
        assertEquals("VertexRaw(pos=[480.0, 272.0, 0.0], tex=[0.0, 0.0, 0.0])", readOne())
        assertEquals("VertexRaw(pos=[0.0, 272.0, 0.0], tex=[0.0, 0.0, 0.0])", readOne())
        assertEquals("VertexRaw(pos=[480.0, 0.0, 0.0], tex=[0.0, 0.0, 0.0])", readOne())
    }
}