package com.soywiz.dynarek2

import kotlin.test.*

open class D2IntegrationTest {
    open fun testSimple(expected: Int, iregs: IntArray = intArrayOf(), gen: D2Builder.() -> Unit) {
        NewD2Memory(512 * 4) { mem ->
            for (n in iregs.indices) mem.set32(n, iregs[n])
            val result = D2Func { gen() }.generate().func(mem, null, null, null)
            assertEquals(expected, result)
        }
    }

    @Test
    fun testSimple() {
        testSimple(3) { RETURN(1.lit + 2.lit) }
        testSimple(7) { RETURN(1.lit + (2.lit * 3.lit)) }
    }

    @Test
    fun testRegs() {
        val p0 = 0
        val p1 = 1
        testSimple(8, iregs = intArrayOf(8)) { RETURN(REGS32(p0)) }
        testSimple(8, iregs = intArrayOf(3, 5)) { RETURN(REGS32(p0) + REGS32(p1)) }
    }

    @Test
    fun testCond() {
        testSimple(FALSE, iregs = intArrayOf(0)) { RETURN(REGS32(0) GT 0.lit) }
        testSimple(TRUE, iregs = intArrayOf(1)) { RETURN(REGS32(0) GT 0.lit) }
    }

    @Test
    fun testIf() {
        val p0 = 0

        fun D2Builder.myif() {
            IF(REGS32(p0)) {
                RETURN(10.lit)
            }
            RETURN(20.lit)
        }

        fun D2Builder.myifElse() {
            IF(REGS32(p0)) {
                RETURN(10.lit)
            } ELSE {
                RETURN(20.lit)
            }
        }

        testSimple(10, iregs = intArrayOf(TRUE)) { myif() }
        testSimple(20, iregs = intArrayOf(FALSE)) { myif() }
        testSimple(10, iregs = intArrayOf(TRUE)) { myifElse() }
        testSimple(20, iregs = intArrayOf(FALSE)) { myifElse() }
    }

    @Test
    fun testWhile() {
        val counterIndex = 0
        val outIndex = 1

        testSimple(277, iregs = intArrayOf(100, 77)) {
            val counter = REGS32(counterIndex)
            val out = REGS32(outIndex)

            WHILE(counter GT 0.lit) {
                SET(out, out + 2.lit)
                SET(counter, counter - 1.lit)
                //PRINTI(counter)
            }
            RETURN(counter + out)
        }
    }

    private val TRUE = 1
    private val FALSE = 0
}