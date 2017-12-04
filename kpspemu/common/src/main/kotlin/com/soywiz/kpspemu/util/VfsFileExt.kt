package com.soywiz.kpspemu.util

import com.soywiz.korio.error.ignoreErrors
import com.soywiz.korio.vfs.VfsFile

// Shouldn't be necessary
suspend fun VfsFile.mkdirsSafe() = ignoreErrors { mkdirs() }
