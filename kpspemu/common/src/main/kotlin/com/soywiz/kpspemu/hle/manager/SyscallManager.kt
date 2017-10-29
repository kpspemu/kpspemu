package com.soywiz.kpspemu.hle.manager

import com.soywiz.kpspemu.hle.modules.NativeFunction

class SyscallManager {
	var lasSyscallId = 1

	val syscallToFunc = LinkedHashMap<Int, NativeFunction>()

	fun register(nfunc: NativeFunction): Int {
		val syscallId = lasSyscallId++
		syscallToFunc[syscallId] = nfunc
		return syscallId
	}
}