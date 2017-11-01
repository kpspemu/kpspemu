package com.soywiz.kpspemu.hle.manager

import com.soywiz.korio.stream.AsyncStream
import com.soywiz.korio.vfs.VfsFile
import com.soywiz.korio.vfs.VfsUtil
import com.soywiz.kpspemu.Emulator
import com.soywiz.kpspemu.util.ResourceItem
import com.soywiz.kpspemu.util.ResourceList

class FileManager(val emulator: Emulator) {
	val deviceManager get() = emulator.deviceManager
	var currentDirectory = "umd0:/"

	val fileDescriptors = ResourceList<FileDescriptor>("FileDescriptor") { FileDescriptor(it) }

	fun resolvePath(path: String): String {
		if (path.contains(':')) {
			return path
		} else {
			if (path.startsWith('/')) {
				return currentDirectory.split(':').first() + ":" + path
			} else {
				return VfsUtil.combine(currentDirectory, path)
			}
		}
	}
	fun resolve(path: String): VfsFile {
		val resolvedPath = resolvePath(path)
		//println("resolvedPath --> $resolvedPath")
		return deviceManager.root[resolvedPath]
	}
}

class FileDescriptor(override val id: Int) : ResourceItem {
	lateinit var file: VfsFile
	lateinit var stream: AsyncStream
}
