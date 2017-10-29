package com.soywiz.kpspemu.hle.modules

import com.soywiz.korma.random.MtRand
import com.soywiz.kpspemu.hle.SceModule
import com.soywiz.kpspemu.mem.Memory

class UtilsForUser : SceModule("UtilsForUser") {
	//val random = MtRand(KorioNative.currentTimeMillis())
	val random = MtRand(0)

	override fun registerModule() {
		registerFunctionInt("sceKernelUtilsMt19937Init", uid = 0xE860E75E, since = 150, syscall = 0x20BF) { sceKernelUtilsMt19937Init(mem, int, int) }
		registerFunctionInt("sceKernelUtilsMt19937UInt", uid = 0x06FB8A63, since = 150, syscall = 0x20C0) { sceKernelUtilsMt19937UInt(mem, int) }
	}

	fun sceKernelUtilsMt19937Init(memory: Memory, ctx: Int, seed: Int): Int {
		println("Not implemented UtilsForUser.sceKernelUtilsMt19937Init")
		return 0
	}

	fun sceKernelUtilsMt19937UInt(memory: Memory, ctx: Int): Int {
		//println("Not implemented UtilsForUser.sceKernelUtilsMt19937UInt")
		val value = random.nextInt()
		//println("Random: $value")
		return value
	}
}