package com.soywiz.kpspemu.hle.manager

import com.soywiz.kpspemu.Emulator
import com.soywiz.kpspemu.mem.Ptr

class CallbackManager(emulator: Emulator) : Manager<PspCallback>("Callback", emulator, initialId = 1) {
	fun create(name: String, func: Ptr, arg: Int): PspCallback = PspCallback(this, allocId(), name, func, arg)

	fun queueFunction1(funcPC: Int, funcARG: Int) {
		// @TODO: Implement this!
	}

	override fun reset() {
		super.reset()
	}
}

class PspCallback(
	val callbackManager: CallbackManager,
	id: Int,
	name: String,
	val func: Ptr,
	val arg: Int
) : Resource(callbackManager, id, name){

}