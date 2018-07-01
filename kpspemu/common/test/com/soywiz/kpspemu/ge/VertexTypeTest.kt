package com.soywiz.kpspemu.ge

import com.soywiz.kpspemu.*
import kotlin.test.*

class VertexTypeTest : BaseTest() {
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
            "VertexType(color=COLOR8888, normal=VOID, pos=SHORT, tex=VOID, weight=VOID, size=12)",
            VertexType(0x0080011C).toString()
        )
    }
}