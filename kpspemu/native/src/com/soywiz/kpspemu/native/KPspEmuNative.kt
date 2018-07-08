package com.soywiz.kpspemu.native

actual object KPspEmuNative {
    actual fun getCurrentDirectory(): String = "."
    actual val documentLocationHash: String = "#"
    actual fun initialization(): Unit {
    }

    actual fun invalidateCache() {
    }
}
