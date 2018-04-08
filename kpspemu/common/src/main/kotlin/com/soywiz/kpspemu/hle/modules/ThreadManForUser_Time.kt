package com.soywiz.kpspemu.hle.modules

import com.soywiz.kpspemu.*
import com.soywiz.kpspemu.hle.*
import com.soywiz.kpspemu.hle.error.*
import com.soywiz.kpspemu.mem.*

class ThreadManForUser_Time(val tmodule: ThreadManForUser) : SceSubmodule<ThreadManForUser>(tmodule) {
    fun sceKernelGetSystemTimeWide(): Long = rtc.getTimeInMicroseconds()
    fun sceKernelGetSystemTimeLow(): Int = rtc.getTimeInMicrosecondsInt()
    fun sceKernelGetSystemTime(ptr: Ptr64): Int {
        if (ptr.isNull) return SceKernelErrors.ERROR_ERRNO_INVALID_ARGUMENT
        ptr[0] = rtc.getTimeInMicroseconds()
        return 0
    }

    fun registerSubmodule() = tmodule.apply {
        registerFunctionLong("sceKernelGetSystemTimeWide", 0x82BC5777, since = 150) { sceKernelGetSystemTimeWide() }
        registerFunctionInt("sceKernelGetSystemTimeLow", 0x369ED59D, since = 150) { sceKernelGetSystemTimeLow() }
        registerFunctionInt("sceKernelGetSystemTime", 0xDB738F35, since = 150) { sceKernelGetSystemTime(ptr64) }
    }
}
