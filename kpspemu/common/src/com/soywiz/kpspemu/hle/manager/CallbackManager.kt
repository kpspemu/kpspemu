package com.soywiz.kpspemu.hle.manager

import com.soywiz.kpspemu.*
import com.soywiz.kpspemu.mem.*

class CallbackManager(emulator: Emulator) : Manager<PspCallback>("Callback", emulator, initialId = 1) {
    fun create(name: String, func: Ptr, arg: Int): PspCallback = PspCallback(this, allocId(), name, func, arg)

    override fun reset() {
        super.reset()
    }

    fun queueFunction1(funcPC: Int, arg1: Int) {
        // @TODO: Implement this!
    }

    fun queueCallback(callback: PspCallback, arg2: Int) {
        // @TODO: Implement this!
    }

    fun cancelCallback(callback: PspCallback) {
        // @TODO: Implement this!
    }

    fun executeCallbacks() {
        // @TODO: Implement this!
    }
}

class PspCallback(
    val callbackManager: CallbackManager,
    id: Int,
    name: String,
    val func: Ptr,
    val arg: Int
) : Resource(callbackManager, id, name)
