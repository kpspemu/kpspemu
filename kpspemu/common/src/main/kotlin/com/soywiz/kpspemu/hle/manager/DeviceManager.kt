package com.soywiz.kpspemu.hle.manager

import com.soywiz.korio.vfs.VfsFile
import com.soywiz.kpspemu.Emulator

class DeviceManager(val emulator: Emulator) {
	val devicesToVfs = LinkedHashMap<String, VfsFile>()

	/*
		fileManager.mount('fatms0', msvfs);
		fileManager.mount('ms0', msvfs);
		fileManager.mount('mscmhc0', msvfs);

		fileManager.mount('host0', new MemoryVfs());
		fileManager.mount('flash0', new UriVfs('data/flash0'));
		fileManager.mount('emulator', this.emulatorVfs);
		fileManager.mount('kemulator', this.emulatorVfs);

		fileManager.mount('disc0', mount);
		fileManager.mount('umd0', mount);
	 */
}