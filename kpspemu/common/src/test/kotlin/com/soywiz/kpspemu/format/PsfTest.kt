package com.soywiz.kpspemu.format

import KpspTests
import com.soywiz.korio.async.syncTest
import mytest.assertEquals
import org.junit.Test

class PsfTest {
    @Test
    fun name() = syncTest {
        val psf = Psf.fromStream(KpspTests.rootTestResources["controller.sfo"].readAsSyncStream())
        assertEquals(
            "[Entry(BOOTABLE, 1), Entry(CATEGORY, MG), Entry(REGION, 32768), Entry(TITLE, Basic controller sample)]",
            psf.entries.toString()
        )
    }
}