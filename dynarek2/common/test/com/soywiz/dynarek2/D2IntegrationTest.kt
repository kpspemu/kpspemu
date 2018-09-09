package com.soywiz.dynarek2

import kotlin.test.*

open class D2IntegrationTest {
    val allDebug = true
    //val allDebug = false
    val context = D2Context()

    open fun testSimple(expected: Int, iregs: IntArray = intArrayOf(), name: String? = null, debug: Boolean? = null, gen: D2Builder.() -> Unit) {
        NewD2Memory(512 * 4) { mem ->
            for (n in iregs.indices) mem.set32(n, iregs[n])
            val result = D2Func { gen() }.generate(context, name, debug ?: allDebug).func(mem, null, null, null)
            assertEquals(expected, result)
        }
    }

    @Test
    fun testSimple() {
        testSimple(1 + 2, name = "simple0") { RETURN(1.lit + 2.lit) }
        testSimple(1 + (2 * 3), name = "simple1") { RETURN(1.lit + (2.lit * 3.lit)) }
        testSimple((4 * 5) + (2 * 3), name = "simple2") { RETURN((4.lit * 5.lit) + (2.lit * 3.lit)) }
        testSimple(((4 * 5) * 7) + (2 * 3), name = "simple3") { RETURN(((4.lit * 5.lit) * 7.lit) + (2.lit * 3.lit)) }
        testSimple((4 * 5) + ((2 * 3) * 7), name = "simple4") { RETURN(((4.lit * 5.lit) + ((2.lit * 3.lit) * 7.lit))) }
        testSimple(((4 * 5) + (2 * 3)) * 7, name = "simple5") { RETURN(((4.lit * 5.lit) + (2.lit * 3.lit)) * 7.lit) }
    }

    @Test
    fun testShift() {
        testSimple(1 shl 8, name = "shift0") { RETURN(1.lit SHL 8.lit) }
        testSimple((-1) shr 8, name = "shift1") { RETURN((-1).lit SHR 8.lit) }
        testSimple((-1) ushr 8, name = "shift2") { RETURN((-1).lit USHR 8.lit) }
    }

    @Test
    fun testLogic() {
        val A = 0x7f7f7f1f
        val B = 0x33F333F3
        testSimple(A and B, name = "logic0") { RETURN(A.lit AND B.lit) }
        testSimple(A or B, name = "logic1") { RETURN(A.lit OR B.lit) }
        testSimple(A xor B, name = "logic2") { RETURN(A.lit XOR B.lit) }
    }

    @Test
    fun testRegs() {
        val p0 = 0
        val p1 = 1
        testSimple(8, iregs = intArrayOf(8), name = "regs0", debug = true) { RETURN(REGS32(p0)) }
        testSimple(8, iregs = intArrayOf(3, 5), name = "regs1", debug = true) { RETURN(REGS32(p0) + REGS32(p1)) }
    }

    @Test
    fun testCond() {
        testSimple(FALSE, name = "cond0", iregs = intArrayOf(0)) { RETURN(REGS32(0) GT 0.lit) }
        testSimple(TRUE, name = "cond1", iregs = intArrayOf(1)) { RETURN(REGS32(0) GT 0.lit) }
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

        testSimple(10, iregs = intArrayOf(TRUE), name = "if0") { myif() }
        testSimple(20, iregs = intArrayOf(FALSE), name = "if1") { myif() }
        testSimple(10, iregs = intArrayOf(TRUE), name = "if2") { myifElse() }
        testSimple(20, iregs = intArrayOf(FALSE), name = "if3") { myifElse() }
    }

    @Test
    fun testWhile() {
        val counterIndex = 0
        val outIndex = 1

        testSimple(277, iregs = intArrayOf(100, 77), name = "while0") {
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