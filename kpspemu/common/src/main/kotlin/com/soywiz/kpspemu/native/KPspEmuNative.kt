package com.soywiz.kpspemu.native

import com.soywiz.korio.vfs.LocalVfs

expect object KPspEmuNative {
	fun getCurrentDirectory(): String
}

val CurrentVfs by lazy { LocalVfs(KPspEmuNative.getCurrentDirectory()) }