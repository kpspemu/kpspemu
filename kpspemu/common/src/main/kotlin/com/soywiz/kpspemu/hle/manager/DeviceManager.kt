package com.soywiz.kpspemu.hle.manager

import com.soywiz.korio.vfs.MemoryVfs
import com.soywiz.korio.vfs.PathInfo
import com.soywiz.korio.vfs.VfsFile
import com.soywiz.korio.vfs.VfsUtil
import com.soywiz.kpspemu.Emulator
import com.soywiz.kpspemu.util.MountableSync
import com.soywiz.kpspemu.util.MountableVfsSync

class DeviceManager(val emulator: Emulator) {
	var currentDirectory = "ms0:/PSP/GAME/app"
	val dummy = MemoryVfs()
	val root = MountableVfsSync {
		mount("fatms0:", dummy)
		mount("ms0:", dummy)
		mount("mscmhc0:", dummy)

		mount("host0:", dummy)
		mount("flash0:", dummy)
		mount("emulator:", dummy)
		mount("kemulator:", dummy)

		mount("disc0:", dummy)
		mount("umd0:", dummy)
	}
	val mountable = root.vfs as MountableSync

	//val devicesToVfs = LinkedHashMap<String, VfsFile>()

	fun mount(name: String, vfs: VfsFile) {
		mountable.mount(name, vfs)
		//devicesToVfs[name] = vfs
	}

	fun resolve(path: String): VfsFile {
		if (path.contains(':')) {
			return root[path]
		} else {
			return root[VfsUtil.combine(currentDirectory, path)]
		}
	}
}