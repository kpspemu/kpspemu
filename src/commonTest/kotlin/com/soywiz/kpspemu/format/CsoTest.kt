package com.soywiz.kpspemu.format

import com.soywiz.korio.file.std.*
import com.soywiz.korio.stream.*
import com.soywiz.kpspemu.*
import kotlin.test.*

class CsoTest : BaseTest() {
    @Test
    fun name() = pspSuspendTest {
        val csoFile = resourcesVfs["cube.cso"]
        val isoFile = resourcesVfs["cube.iso"]
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