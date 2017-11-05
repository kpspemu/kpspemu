package com.soywiz.kpspemu.native

import kotlin.browser.document

actual object KPspEmuNative {
	actual fun getCurrentDirectory(): String = "."
	actual val documentLocationHash: String get() = document.location?.hash ?: "#"
}
