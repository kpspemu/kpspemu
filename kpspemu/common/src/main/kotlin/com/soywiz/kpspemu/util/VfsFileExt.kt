package com.soywiz.kpspemu.util

import com.soywiz.korio.error.*
import com.soywiz.korio.vfs.*

// Shouldn't be necessary
suspend fun VfsFile.mkdirsSafe() = ignoreErrors { mkdirs() }
