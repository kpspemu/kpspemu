package com.soywiz.kpspemu.native

import java.io.File

actual object KPspEmuNative {
    actual fun getCurrentDirectory(): String = File(".").absolutePath
    actual val documentLocationHash: String = "#"
}
