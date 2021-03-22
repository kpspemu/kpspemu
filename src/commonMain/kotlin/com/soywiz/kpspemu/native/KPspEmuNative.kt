package com.soywiz.kpspemu.native

import com.soywiz.korio.file.std.*

expect object KPspEmuNative {
    fun getCurrentDirectory(): String
    fun initialization(): Unit
    fun invalidateCache()
    val documentLocationHash: String
}

val CurrentVfs by lazy { localVfs(KPspEmuNative.getCurrentDirectory()) }
