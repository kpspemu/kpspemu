package com.soywiz.dynarek2

import kotlin.test.*

open class D2IntegrationTest {
    val allDebug = true
    //val allDebug = false
    val context = D2Context()

    open fun testSimple(
        expected: Int,
        iregs: IntArray = intArrayOf(),
        name: String? = null,
        debug: Boolean? = null,
        external: Any? = null,
        checks: (regs: D2Memory) -> Unit = {},
        gen: D2Builder.() -> Unit
    ) {
        NewD2Memory(512 * 4) { mem ->
            for (n in iregs.indices) mem.set32(n, iregs[n])
            val runner = D2Runner()
            try {
                runner.setParams(mem, null, null, external)
                runner.setFunc(D2Func { gen() }.generate(context, name, debug ?: allDebug))
                val actual = runner.execute()
                assertEquals(expected, actual, "Expected <$expected>, actual <$actual> (for function '$name')")
                checks(mem)
            } finally {
                runner.close()
            }
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
    fun testArithmetic() {
        val L = 77
        val R = 3

        testSimple(L + R, name = "arith_add") { RETURN(L.lit + R.lit) }
        testSimple(L - R, name = "arith_sub") { RETURN(L.lit - R.lit) }
        testSimple(L * R, name = "arith_mul") { RETURN(L.lit * R.lit) }
        testSimple(L / R, name = "arith_div") { RETURN(L.lit / R.lit) }
        testSimple(L % R, name = "arith_rem") { RETURN(L.lit % R.lit) }
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
    fun testIfFixed() {
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

        testSimple(10, iregs = intArrayOf(TRUE), name = "if_fixed_0") { myif() }
        testSimple(20, iregs = intArrayOf(FALSE), name = "if_fixed_1") { myif() }
        testSimple(10, iregs = intArrayOf(TRUE), name = "if_fixed_else_0") { myifElse() }
        testSimple(20, iregs = intArrayOf(FALSE), name = "if_fixed_else_1") { myifElse() }
    }

    @Test
    fun testIfComp() {
        val p0 = 0

        fun D2Builder.myif() {
            IF(REGS32(p0) GT 0.lit) {
                RETURN(10.lit)
            }
            RETURN(20.lit)
        }

        fun D2Builder.myifElse() {
            IF(REGS32(p0) GT 0.lit) {
                RETURN(10.lit)
            } ELSE {
                RETURN(20.lit)
            }
        }

        testSimple(10, iregs = intArrayOf(1), name = "if_comp_0") { myif() }
        testSimple(20, iregs = intArrayOf(0), name = "if_comp_1") { myif() }
        testSimple(20, iregs = intArrayOf(-1), name = "if_comp_2") { myif() }
        testSimple(10, iregs = intArrayOf(1), name = "if_comp_else_0") { myifElse() }
        testSimple(20, iregs = intArrayOf(0), name = "if_comp_else_1") { myifElse() }
        testSimple(20, iregs = intArrayOf(-1), name = "if_comp_else_2") { myifElse() }
    }

    @Test
    fun testSet32Fixed() {
        testSimple(7, iregs = intArrayOf(100, 77), name = "set32_fixed0", gen = {
            SET(REGS32(1), 7.lit)
            RETURN(REGS32(1))
        }, checks = { mem ->
            assertEquals(7, mem.get32(1))
        })
    }

    @Test
    fun testSet8Fixed() {
        testSimple(0, iregs = intArrayOf(), name = "set8_fixed0", gen = {
            SET(D2Expr.RefI(D2MemSlot.REGS, D2Size.BYTE, 3.lit), 3.lit)
            SET(D2Expr.RefI(D2MemSlot.REGS, D2Size.BYTE, 2.lit), 2.lit)
            SET(D2Expr.RefI(D2MemSlot.REGS, D2Size.BYTE, 1.lit), 1.lit)
            SET(D2Expr.RefI(D2MemSlot.REGS, D2Size.BYTE, 0.lit), 0.lit)
            RETURN(0.lit)
        }, checks = { mem ->
            assertEquals(0, mem.get8(0))
            assertEquals(1, mem.get8(1))
            assertEquals(2, mem.get8(2))
            assertEquals(3, mem.get8(3))
            assertEquals(0, mem.get8(4))
        })
    }

    @Test
    fun testSet16Fixed() {
        testSimple(0, iregs = intArrayOf(), name = "set16_fixed0", gen = {
            SET(D2Expr.RefI(D2MemSlot.REGS, D2Size.SHORT, 1.lit), 1.lit)
            SET(D2Expr.RefI(D2MemSlot.REGS, D2Size.SHORT, 0.lit), 0.lit)
            RETURN(0.lit)
        }, checks = { mem ->
            assertEquals(0, mem.get16(0))
            assertEquals(1, mem.get16(1))
        })
    }

    @Test
    fun testSet32Fixed_2() {
        testSimple(0, iregs = intArrayOf(), name = "set32_fixed0", gen = {
            SET(D2Expr.RefI(D2MemSlot.REGS, D2Size.INT, 1.lit), 1.lit)
            SET(D2Expr.RefI(D2MemSlot.REGS, D2Size.INT, 0.lit), 0.lit)
            RETURN(0.lit)
        }, checks = { mem ->
            assertEquals(0, mem.get32(0))
            assertEquals(1, mem.get32(1))
        })
    }

    @Test
    fun testSet8Dynamic() {
        testSimple(0 + 1 + 2 + 3, iregs = intArrayOf(), name = "set8_dynamic0", gen = {
            val counter = REGS32(16)
            val out = REGS32(17)
            SET(out, 0.lit)
            FOR(counter, 0.lit, 4.lit) {
                val ref = D2Expr.RefI(D2MemSlot.REGS, D2Size.BYTE, counter)
                SET(ref, counter)
                SET(out, out + ref)
            }
            RETURN(out)
        }, checks = { mem ->
            assertEquals(0, mem.get8(0))
            assertEquals(1, mem.get8(1))
            assertEquals(2, mem.get8(2))
            assertEquals(3, mem.get8(3))
            assertEquals(0, mem.get8(4))
        })
    }

    @Test
    fun testSet16Dynamic() {
        testSimple(0 + 1 + 2 + 3, iregs = intArrayOf(), name = "set16_dynamic0", gen = {
            val counter = REGS32(16)
            val out = REGS32(17)
            SET(out, 0.lit)
            FOR(counter, 0.lit, 4.lit) {
                val ref = D2Expr.RefI(D2MemSlot.REGS, D2Size.SHORT, counter)
                SET(ref, counter)
                SET(out, out + ref)
            }
            RETURN(out)
        }, checks = { mem ->
            assertEquals(0, mem.get16(0))
            assertEquals(1, mem.get16(1))
            assertEquals(2, mem.get16(2))
            assertEquals(3, mem.get16(3))
            assertEquals(0, mem.get16(4))
        })
    }

    @Test
    fun testSet32Dynamic_2() {
        testSimple(0, iregs = intArrayOf(), name = "set32_dynamic0", gen = {
            val counter = REGS32(16)
            FOR(counter, 0.lit, 4.lit) {
                SET(D2Expr.RefI(D2MemSlot.REGS, D2Size.INT, counter), counter)
            }
            RETURN(0.lit)
        }, checks = { mem ->
            assertEquals(0, mem.get32(0))
            assertEquals(1, mem.get32(1))
            assertEquals(2, mem.get32(2))
            assertEquals(3, mem.get32(3))
            assertEquals(0, mem.get32(4))
        })
    }

    @Test
    fun testGet8() {
        testSimple(0x04, iregs = intArrayOf(0x01020304), name = "get8_0") {
            RETURN(
                D2Expr.RefI(
                    D2MemSlot.REGS,
                    D2Size.BYTE,
                    0.lit
                )
            )
        }
        testSimple(0x03, iregs = intArrayOf(0x01020304), name = "get8_1") {
            RETURN(
                D2Expr.RefI(
                    D2MemSlot.REGS,
                    D2Size.BYTE,
                    1.lit
                )
            )
        }
    }

    @Test
    fun testIncrementFixed() {
        testSimple(199, iregs = intArrayOf(100, 100), name = "increment_fixed_0", gen = {
            SET(REGS32(0), REGS32(0) + 99.lit)
            //IPRINT(REGS32(1) + 77.lit)
            SET(REGS32(1), REGS32(1) + 77.lit)
            RETURN(REGS32(0))
        }, checks = { mem ->
            assertEquals(-199, -mem.get32(0))
            assertEquals(-177, -mem.get32(1))
        })
    }

    @Test
    fun testSet32Dynamic() {
        testSimple(7, iregs = intArrayOf(100, 77), name = "set32_dynamic0", gen = {
            SET(REGS32(0), 1.lit)
            SET(REGS32(REGS32(0)), 7.lit)
            RETURN(REGS32(REGS32(0)))
        }, checks = { mem ->
            assertEquals(7, mem.get32(1))
        })
    }

    @Test
    fun testWhile() {
        val counterIndex = 0
        val outIndex = 1

        testSimple(277, iregs = intArrayOf(100, 77), name = "while0") {
            val counter = REGS32(counterIndex)
            val out = REGS32(outIndex)

            //IPRINT(counter)
            WHILE(counter GT 0.lit) {
                SET(out, out + 2.lit)
                SET(counter, counter - 1.lit)
                //IPRINT(999999.lit)
                //IPRINT(out)
                //IPRINT(counter)
            }
            RETURN(counter + out)
        }
    }

    @Test
    fun testCall() {
        testSimple(3, name = "call0") {
            IPRINT(::isqrt.invoke(9.lit))
            RETURN(::isqrt.invoke(9.lit))
        }
        testSimple(1, name = "call1") { RETURN(::isub.invoke(3.lit, 2.lit)) }
    }

    @Test
    fun testFArith() {
        testSimple(0, name = "fadd0", gen = { SET(REGF32(0), 0.5f.lit + 1.0f.lit); RETURN(0.lit) }, checks = { mem ->
            assertEquals(1.5f, mem.getF32(0))
        })
        testSimple(0, name = "fsub0", gen = { SET(REGF32(0), 0.5f.lit - 1.0f.lit); RETURN(0.lit) }, checks = { mem ->
            assertEquals(-0.5f, mem.getF32(0))
        })
        testSimple(0, name = "fmul0", gen = { SET(REGF32(0), 2.0f.lit * 4.0f.lit); RETURN(0.lit) }, checks = { mem ->
            assertEquals(8.0f, mem.getF32(0))
        })
        testSimple(0, name = "fdiv0", gen = { SET(REGF32(0), 2.0f.lit / 4.0f.lit); RETURN(0.lit) }, checks = { mem ->
            assertEquals(0.5f, mem.getF32(0))
        })
    }

    @Test
    fun testCallFloat() {
        testSimple(
            0,
            name = "fcall0",
            gen = { SET(REGF32(0), ::fsub.invoke(0.5f.lit, 1.0f.lit)); RETURN(0.lit) },
            checks = { mem ->
                assertEquals(-0.5f, mem.getF32(0))
            })
    }

    // @TODO: Can't throw exceptions inside dynarek function

    //@Test
    //fun testCallThrow() {
    //    expectException<MyDemoException> {
    //        testSimple(0, name = "call0") {
    //            RETURN(::demo_ithrow.invoke(*arrayOf()))
    //        }
    //    }
    //    /*
    //    try {
    //        testSimple(0, name = "call0") {
    //            RETURN(::demo_ithrow.invoke(*arrayOf()))
    //        }
    //        fail("EXPECTED EXCEPTION")
    //    } catch (e: MyDemoException) {
    //        // worked fine
    //    } catch (e: Throwable) {
    //        e.printStackTrace()
    //        throw e
    //    }
    //    */
    //}

    /*
    @Test
    fun testCallFloat() {
        val reg = D2ExprBuilder { REGF32(1) }
        val value = 0.5f
        testSimple(0, name = "callFloat0", gen = {
            SET(reg, value.lit)
            RETURN(0.lit)
        }, checks = { mem ->
            assertEquals(value, mem.getF32(1))
        })
    }
    */

    @Test
    fun testCallExternal() {
        testSimple(7, name = "external0", external = MyClass()) {
            RETURN(::dyna_getDemo.invoke(EXTERNAL))
        }
    }

    private val TRUE = 1
    private val FALSE = 0
}
