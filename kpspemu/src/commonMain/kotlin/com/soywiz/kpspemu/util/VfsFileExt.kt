package com.soywiz.kpspemu.util

import com.soywiz.korio.error.*
import com.soywiz.korio.file.*

// Shouldn't be necessary
suspend fun VfsFile.mkdirsSafe() = ignoreErrors { mkdir() }
