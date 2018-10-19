package com.soywiz.kpspemu.hle.modules

import com.soywiz.kpspemu.*
import com.soywiz.kpspemu.hle.*
import com.soywiz.kpspemu.hle.error.*
import com.soywiz.kpspemu.mem.*

class ThreadManForUser_Callback(val tmodule: ThreadManForUser) : SceSubmodule<ThreadManForUser>(tmodule) {
    private fun callback(id: Int) =
        callbackManager.tryGetById(id) ?: sceKernelException(SceKernelErrors.ERROR_KERNEL_NOT_FOUND_CALLBACK)

    fun sceKernelCreateCallback(name: String?, func: Ptr, arg: Int): Int {
        val callback = callbackManager.create(name ?: "callback", func, arg)
        return callback.id
    }

    fun sceKernelCheckCallback(): Int {
        callbackManager.executeCallbacks()
        return 0
    }

    fun sceKernelDeleteCallback(id: Int): Int {
        callbackManager.freeById(id)
        return 0
    }

    fun sceKernelNotifyCallback(cbid: Int, arg2: Int): Int {
        callbackManager.queueCallback(callback(cbid), arg2)
        return 0
    }

    fun sceKernelCancelCallback(cbid: Int): Int {
        callbackManager.cancelCallback(callback(cbid))
        return 0
    }

    fun registerSubmodule() = tmodule.apply {
        registerFunctionInt("sceKernelCreateCallback", 0xE81CAF8F, since = 150) {
            sceKernelCreateCallback(
                str,
                ptr,
                int
            )
        }
        registerFunctionInt("sceKernelCheckCallback", 0x349D6D6C, since = 150) { sceKernelCheckCallback() }
        registerFunctionInt("sceKernelDeleteCallback", 0xEDBA5844, since = 150) { sceKernelDeleteCallback(int) }
        registerFunctionInt("sceKernelNotifyCallback", 0xC11BA8C4, since = 150) { sceKernelNotifyCallback(int, int) }
        registerFunctionInt("sceKernelCancelCallback", 0xBA4051D6, since = 150) { sceKernelCancelCallback(int) }
    }
}
