package com.soywiz.kpspemu.hle.modules

import com.soywiz.kpspemu.cpu.*
import com.soywiz.kpspemu.hle.*

class ThreadManForUser_Alarm(val tmodule: ThreadManForUser) : SceSubmodule<ThreadManForUser>(tmodule) {
    fun sceKernelSetAlarm(cpu: CpuState): Unit = UNIMPLEMENTED(0x6652B8CA)
    fun sceKernelCancelAlarm(cpu: CpuState): Unit = UNIMPLEMENTED(0x7E65B999)
    fun sceKernelSetSysClockAlarm(cpu: CpuState): Unit = UNIMPLEMENTED(0xB2C25152)
    fun sceKernelReferAlarmStatus(cpu: CpuState): Unit = UNIMPLEMENTED(0xDAA3F564)

    fun registerSubmodule() = tmodule.apply {
        registerFunctionRaw("sceKernelSetAlarm", 0x6652B8CA, since = 150) { sceKernelSetAlarm(it) }
        registerFunctionRaw("sceKernelCancelAlarm", 0x7E65B999, since = 150) { sceKernelCancelAlarm(it) }
        registerFunctionRaw("sceKernelSetSysClockAlarm", 0xB2C25152, since = 150) { sceKernelSetSysClockAlarm(it) }
        registerFunctionRaw("sceKernelReferAlarmStatus", 0xDAA3F564, since = 150) { sceKernelReferAlarmStatus(it) }
    }
}
