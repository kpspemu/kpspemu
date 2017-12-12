package com.soywiz.kpspemu.hle.modules

import com.soywiz.kpspemu.callbackManager
import com.soywiz.kpspemu.cpu.CpuState
import com.soywiz.kpspemu.hle.SceSubmodule
import com.soywiz.kpspemu.mem.Ptr

class ThreadManForUser_Callback(val tmodule: ThreadManForUser) : SceSubmodule<ThreadManForUser>(tmodule) {
	fun sceKernelCreateCallback(name: String?, func: Ptr, arg: Int): Int {
		val callback = callbackManager.create(name ?: "callback", func, arg)
		return callback.id
	}

	fun sceKernelCheckCallback(): Int {
		// TODO
		return 0
	}

	fun sceKernelDeleteCallback(id: Int): Int {
		callbackManager.freeById(id)
		return 0
	}

	fun sceKernelCancelCallback(cpu: CpuState): Unit = UNIMPLEMENTED(0xBA4051D6)
	fun sceKernelNotifyCallback(cpu: CpuState): Unit = UNIMPLEMENTED(0xC11BA8C4)

	fun registerSubmodule() = tmodule.apply {
		registerFunctionInt("sceKernelCreateCallback", 0xE81CAF8F, since = 150) { sceKernelCreateCallback(str, ptr, int) }
		registerFunctionInt("sceKernelCheckCallback", 0x349D6D6C, since = 150) { sceKernelCheckCallback() }
		registerFunctionInt("sceKernelDeleteCallback", 0xEDBA5844, since = 150) { sceKernelDeleteCallback(int) }
		registerFunctionRaw("sceKernelCancelCallback", 0xBA4051D6, since = 150) { sceKernelCancelCallback(it) }
		registerFunctionRaw("sceKernelNotifyCallback", 0xC11BA8C4, since = 150) { sceKernelNotifyCallback(it) }
	}
}
