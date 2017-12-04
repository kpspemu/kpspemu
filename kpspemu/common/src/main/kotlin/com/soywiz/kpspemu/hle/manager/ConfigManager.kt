package com.soywiz.kpspemu.hle.manager

import com.soywiz.korinject.AsyncDependency
import com.soywiz.korio.async.Signal
import com.soywiz.korio.vfs.ApplicationDataVfs
import com.soywiz.korio.vfs.VfsFile
import com.soywiz.kpspemu.util.mkdirsSafe
import com.soywiz.kpspemu.util.nullIf

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