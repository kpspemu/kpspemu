package com.soywiz.kpspemu.hle.manager

import com.soywiz.korinject.*
import com.soywiz.korio.async.*
import com.soywiz.korio.file.*
import com.soywiz.korio.file.std.*
import com.soywiz.kpspemu.util.*

class ConfigManager : AsyncDependency {
    lateinit private var root: VfsFile

    val dropboxBearer: StringConfig by lazy { StringConfig(root["dropbox.bearer"], "") }
    val storage: StringConfig by lazy { StringConfig(root["storage"], "local") }

    suspend override fun init() {
        root = ApplicationDataVfs["config"].apply { mkdirsSafe() }.jail()
    }

    class StringConfig(val file: VfsFile, val default: String) {
        val onChanged = Signal<String>()
        suspend fun get(): String = file.nullIf { !exists() }?.readString() ?: default
        suspend fun set(value: String): Unit = run {
            file.writeString(value)
            onChanged(value)
        }

        suspend fun subscribe(handler: (String) -> Unit) {
            onChanged(handler)
            handler(get())
        }
    }
}