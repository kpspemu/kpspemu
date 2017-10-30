package com.soywiz.kpspemu.hle.manager

import com.soywiz.kpspemu.Emulator
import com.soywiz.kpspemu.mem.Ptr

class CallbackManager(emulator: Emulator) : Manager<PspCallback>(emulator) {
	fun create(name: String, func: Ptr, arg: Int): PspCallback = PspCallback(this, allocId(), name, func, arg)
}

class PspCallback(
	val callbackManager: CallbackManager,
	id: Int,
	name: String,
	val func: Ptr,
	val arg: Int
) : Resource(callbackManager, id, name){

}