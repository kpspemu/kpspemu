package com.soywiz.dynarek2

import java.io.*

class D2IntegrationTestJvm : D2IntegrationTest() {
    override fun testSimple(expected: Int, iregs: IntArray, gen: D2Builder.() -> Unit) {
        try {
            super.testSimple(expected, iregs, gen)
        } catch (e: D2InvalidCodeGen) {
            File("myclass.class").writeBytes(e.data)
            // javap -c ./dynarek2/jvm/myclass.class
            throw e
        }
    }
}

