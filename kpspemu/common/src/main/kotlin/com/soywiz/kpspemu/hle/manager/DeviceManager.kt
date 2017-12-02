package com.soywiz.kpspemu.hle.manager

import com.soywiz.korio.vfs.MemoryVfs
import com.soywiz.korio.vfs.MemoryVfsMix
import com.soywiz.korio.vfs.VfsFile
import com.soywiz.kpspemu.Emulator
import com.soywiz.kpspemu.util.io.MountableSync
import com.soywiz.kpspemu.util.io.MountableVfsSync

class DeviceManager(val emulator: Emulator) {
	val ms = MemoryVfsMix(
		"PSP/GAME/empty.txt" to byteArrayOf(),
		"PSP/COMMON/empty.txt" to byteArrayOf(),
		"PSP/SYSTEM/empty.txt" to byteArrayOf(),
		"PSP/THEME/empty.txt" to byteArrayOf(),
		"PSP/SAVEDATA/empty.txt" to byteArrayOf()
	)
	val flash = MemoryVfsMix(
	)
	val dummy = MemoryVfs()

	val root = MountableVfsSync {

	}
	val mountable = root.vfs as MountableSync

	init {
		reset()
	}

	fun reset() {
		mountable.unmountAll()
		mount("fatms0:", ms)
		mount("ms0:", ms)
		mount("mscmhc0:", ms)
		mount("host0:", dummy)
		mount("flash0:", flash)
		mount("emulator:", dummy)
		mount("kemulator:", dummy)
		mount("disc0:", dummy)
		mount("umd0:", dummy)
	}

	//val devicesToVfs = LinkedHashMap<String, VfsFile>()

	fun mount(name: String, vfs: VfsFile) {
		mountable.unmount(name)
		mountable.mount(name, vfs)
		//devicesToVfs[name] = vfs
	}
}