package com.soywiz.kpspemu.hle.manager

import com.soywiz.korio.vfs.MemoryVfs
import com.soywiz.korio.vfs.VfsFile
import com.soywiz.kpspemu.Emulator
import com.soywiz.kpspemu.util.io.MountableSync
import com.soywiz.kpspemu.util.io.MountableVfsSync

class DeviceManager(val emulator: Emulator) {
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
}