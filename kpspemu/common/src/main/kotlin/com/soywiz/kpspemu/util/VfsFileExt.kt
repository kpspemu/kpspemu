package com.soywiz.kpspemu.util

import com.soywiz.korio.stream.AsyncStream
import com.soywiz.korio.vfs.MemoryVfs
import com.soywiz.korio.vfs.VfsFile

fun AsyncStream.asVfsFile(name: String = "unknown.bin"): VfsFile = MemoryVfs(mapOf(name to this))[name]