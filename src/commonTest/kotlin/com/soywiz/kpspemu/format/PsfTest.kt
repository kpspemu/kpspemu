package com.soywiz.kpspemu.format

import com.soywiz.korio.file.std.*
import com.soywiz.kpspemu.*
import kotlin.test.*

class PsfTest : BaseTest() {
    @Test
    fun name() = pspSuspendTest {
        val psf = Psf.fromStream(resourcesVfs["controller.sfo"].readAsSyncStream())
        assertEquals(
            "[Entry(BOOTABLE, 1), Entry(CATEGORY, MG), Entry(REGION, 32768), Entry(TITLE, Basic controller sample)]",
            psf.entries.toString()
        )
    }
}