package com.soywiz.kpspemu.native

import com.soywiz.korio.vfs.*

expect object KPspEmuNative {
    fun getCurrentDirectory(): String
    fun initialization(): Unit
    fun invalidateCache()
    val documentLocationHash: String
}

val CurrentVfs by lazy { LocalVfs(KPspEmuNative.getCurrentDirectory()) }