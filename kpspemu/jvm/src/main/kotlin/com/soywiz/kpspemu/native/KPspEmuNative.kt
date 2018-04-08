package com.soywiz.kpspemu.native

import java.io.*

actual object KPspEmuNative {
    actual fun getCurrentDirectory(): String = File(".").absolutePath
    actual val documentLocationHash: String = "#"
    actual fun initialization(): Unit {
    }

    actual fun invalidateCache() {
    }
}
