package com.soywiz.kpspemu.hle

import com.soywiz.kpspemu.*
import com.soywiz.kpspemu.hle.manager.*
import kotlin.test.*

class MemoryPartitionTest : BaseTest() {
    val base = MemoryPartition("demo", 0, 100, false)

    @Test
    fun name() {
        assertEquals(100, base.getTotalFreeMemoryInt())
        val p1 = base.allocateLow(10)
        val p2 = base.allocateLow(10)
        val p3 = base.allocateLow(10)
        val p4 = base.allocateLow(10)
        assertEquals(base, p1.root)
        assertEquals(60, base.getTotalFreeMemoryInt())
        assertEquals(60, base.getMaxContiguousFreeMemoryInt())
        p1.deallocate()

        assertEquals(70, base.getTotalFreeMemoryInt())
        assertEquals(60, base.getMaxContiguousFreeMemoryInt())

        p2.deallocate()
        p4.deallocate()

        assertEquals(90, base.getTotalFreeMemoryInt())
        assertEquals(70, base.getMaxContiguousFreeMemoryInt())
    }

    @Test
    fun name2() {
        val p0 = base.allocateSet(30, 20)
        val p1 = base.allocateHigh(10)
        val p2 = base.allocateLow(10)
        assertEquals(20, p0.low)
        assertEquals(90, p1.low)
        assertEquals(0, p2.low)
    }
}