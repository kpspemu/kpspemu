package com.soywiz.kpspemu.format

import KpspTests
import com.soywiz.korio.async.syncTest
import com.soywiz.korio.stream.readAll
import org.junit.Test
import kotlin.test.assertTrue

class CsoTest {
	@Test
	fun name() = syncTest {
		val csoFile = KpspTests.samples["cube.cso"]
		val isoFile = KpspTests.samples["cube.iso"]
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