package com.soywiz.dynarek2

import kotlin.test.*

class D2IntegrationTest {
    @Test
    fun name() {
        val result = D2Func {
            RETURN(1.lit + 2.lit)
        }.generate().func(null, null, null, null)

        assertEquals(3, result)
    }
}