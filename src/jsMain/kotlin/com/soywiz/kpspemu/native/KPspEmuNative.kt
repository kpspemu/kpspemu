package com.soywiz.kpspemu.native

import com.soywiz.korio.*
import kotlinx.browser.*
import org.w3c.dom.*

public external val navigator: dynamic

actual object KPspEmuNative {
    actual fun getCurrentDirectory(): String = "."
    actual fun initialization(): Unit {
        try {
            val screen = window.screen.asDynamic()
            screen.orientation.lock("landscape")
        } catch (e: dynamic) {
            console.error("Couldn't force landscape: ${e.message}")
        }
    }

    actual val documentLocationHash: String get() = document.location?.hash ?: "#"
    actual fun invalidateCache() {
        navigator.serviceWorker.controller.postMessage("refresh");
    }
}
