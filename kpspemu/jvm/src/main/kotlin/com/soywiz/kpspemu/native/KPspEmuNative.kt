package com.soywiz.kpspemu.native

import java.io.File
import java.util.*
import kotlin.reflect.KProperty

actual object KPspEmuNative {
	actual fun getCurrentDirectory(): String = File(".").absolutePath
	actual val documentLocationHash: String = "#"
}
