package com.soywiz.kpspemu.format

import KpspTests
import com.soywiz.korio.async.*
import com.soywiz.korio.stream.*
import org.junit.Test
import kotlin.test.*

class CsoTest {
    @Test
    fun name() = syncTest {
        val csoFile = KpspTests.rootTestResources["cube.cso"]
        val isoFile = KpspTests.rootTestResources["cube.iso"]
        val cso = Cso(csoFile.open())
        //println(cso)
        //println(cso.readCompressedBlock(0).toList())
        //println(cso.readUncompressedBlockCached(0).toList())
        //println(cso.readUncompressedBlockCached(0).toList())
        //println(isoFile.size())
        //println(cso.totalBytes)
        assertTrue(isoFile.readAll().contentEquals(cso.open().readAll()))
    }
}