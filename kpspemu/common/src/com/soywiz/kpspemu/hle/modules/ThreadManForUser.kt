package com.soywiz.kpspemu.hle.modules

import com.soywiz.kpspemu.*
import com.soywiz.kpspemu.cpu.*
import com.soywiz.kpspemu.hle.*
import com.soywiz.kpspemu.util.*

@Suppress("UNUSED_PARAMETER", "MemberVisibilityCanPrivate")
class ThreadManForUser(emulator: Emulator) :
    SceModule(emulator, "ThreadManForUser", 0x40010011, "threadman.prx", "sceThreadManager") {
    val thread = ThreadManForUser_Thread(this)
    val callback = ThreadManForUser_Callback(this)
    val mbx = ThreadManForUser_Mbx(this)
    val sema = ThreadManForUser_Sema(this)
    val mutex = ThreadManForUser_Mutex(this)
    val time = ThreadManForUser_Time(this)
    val eflags = ThreadManForUser_EventFlags(this)
    val fpl = ThreadManForUser_Fpl(this)
    val vpl = ThreadManForUser_Vpl(this)
    val alarm = ThreadManForUser_Alarm(this)

    fun sceKernelGetVTimerTime(cpu: CpuState): Unit = UNIMPLEMENTED(0x034A921F)
    fun _sceKernelReturnFromTimerHandler(cpu: CpuState): Unit = UNIMPLEMENTED(0x0E927AED)
    fun sceKernelUSec2SysClock(cpu: CpuState): Unit = UNIMPLEMENTED(0x110DEC9A)
    fun sceKernelCreateVTimer(cpu: CpuState): Unit = UNIMPLEMENTED(0x20FFF560)
    fun sceKernelGetCallbackCount(cpu: CpuState): Unit = UNIMPLEMENTED(0x2A3D44FF)
    fun ThreadManForUser_31327F19(cpu: CpuState): Unit = UNIMPLEMENTED(0x31327F19)
    fun sceKernelDeleteVTimer(cpu: CpuState): Unit = UNIMPLEMENTED(0x328F9E52)
    fun sceKernelReferMsgPipeStatus(cpu: CpuState): Unit = UNIMPLEMENTED(0x33BE4024)
    fun sceKernelCancelMsgPipe(cpu: CpuState): Unit = UNIMPLEMENTED(0x349B864D)
    fun sceKernelSetVTimerHandlerWide(cpu: CpuState): Unit = UNIMPLEMENTED(0x53B00E9A)
    fun sceKernelSetVTimerTime(cpu: CpuState): Unit = UNIMPLEMENTED(0x542AD630)
    fun sceKernelReferVTimerStatus(cpu: CpuState): Unit = UNIMPLEMENTED(0x5F32BEAA)
    fun sceKernelReferSystemStatus(cpu: CpuState): Unit = UNIMPLEMENTED(0x627E6F3A)
    fun _sceKernelReturnFromCallback(cpu: CpuState): Unit = UNIMPLEMENTED(0x6E9EA350)
    fun ThreadManForUser_71040D5C(cpu: CpuState): Unit = UNIMPLEMENTED(0x71040D5C)
    fun sceKernelReleaseThreadEventHandler(cpu: CpuState): Unit = UNIMPLEMENTED(0x72F3C145)
    fun sceKernelReferCallbackStatus(cpu: CpuState): Unit = UNIMPLEMENTED(0x730ED8BC)
    fun sceKernelReceiveMsgPipe(cpu: CpuState): Unit = UNIMPLEMENTED(0x74829B76)
    fun sceKernelCreateMsgPipe(cpu: CpuState): Unit = UNIMPLEMENTED(0x7C0DC2A0)
    fun sceKernelSendMsgPipeCB(cpu: CpuState): Unit = UNIMPLEMENTED(0x7C41F2C2)
    fun ThreadManForUser_7CFF8CF3(cpu: CpuState): Unit = UNIMPLEMENTED(0x7CFF8CF3)
    fun sceKernelReferGlobalProfiler(cpu: CpuState): Unit = UNIMPLEMENTED(0x8218B4DD)
    fun ThreadManForUser_8672E3D0(cpu: CpuState): Unit = UNIMPLEMENTED(0x8672E3D0)
    fun sceKernelSendMsgPipe(cpu: CpuState): Unit = UNIMPLEMENTED(0x876DBFAD)
    fun sceKernelTrySendMsgPipe(cpu: CpuState): Unit = UNIMPLEMENTED(0x884C9F90)
    fun sceKernelGetVTimerBase(cpu: CpuState): Unit = UNIMPLEMENTED(0xB3A59970)
    fun sceKernelGetVTimerBaseWide(cpu: CpuState): Unit = UNIMPLEMENTED(0xB7C18B77)
    fun sceKernelSysClock2USec(cpu: CpuState): Unit = UNIMPLEMENTED(0xBA6B92E2)
    fun ThreadManForUser_BEED3A47(cpu: CpuState): Unit = UNIMPLEMENTED(0xBEED3A47)
    fun sceKernelGetVTimerTimeWide(cpu: CpuState): Unit = UNIMPLEMENTED(0xC0B3FFD2)
    fun sceKernelStartVTimer(cpu: CpuState): Unit = UNIMPLEMENTED(0xC68D9437)
    fun sceKernelUSec2SysClockWide(cpu: CpuState): Unit = UNIMPLEMENTED(0xC8CD158C)
    fun sceKernelStopVTimer(cpu: CpuState): Unit = UNIMPLEMENTED(0xD0AEEE87)
    fun sceKernelCancelVTimerHandler(cpu: CpuState): Unit = UNIMPLEMENTED(0xD2D615EF)
    fun sceKernelSetVTimerHandler(cpu: CpuState): Unit = UNIMPLEMENTED(0xD8B299AE)
    fun sceKernelTryReceiveMsgPipe(cpu: CpuState): Unit = UNIMPLEMENTED(0xDF52098F)
    fun sceKernelSysClock2USecWide(cpu: CpuState): Unit = UNIMPLEMENTED(0xE1619D7C)
    fun sceKernelDeleteMsgPipe(cpu: CpuState): Unit = UNIMPLEMENTED(0xF0B7DA1C)

    fun sceKernelSetVTimerTimeWide(cpu: CpuState): Unit = UNIMPLEMENTED(0xFB6425C3)
    fun sceKernelReceiveMsgPipeCB(cpu: CpuState): Unit = UNIMPLEMENTED(0xFBFA697D)

    override fun registerModule() {
        thread.registerSubmodule()
        mbx.registerSubmodule()
        callback.registerSubmodule()
        time.registerSubmodule()
        sema.registerSubmodule()
        mutex.registerSubmodule()
        eflags.registerSubmodule()
        fpl.registerSubmodule()
        vpl.registerSubmodule()
        alarm.registerSubmodule()

        registerFunctionRaw("sceKernelGetVTimerTime", 0x034A921F, since = 150) { sceKernelGetVTimerTime(it) }
        registerFunctionRaw(
            "_sceKernelReturnFromTimerHandler",
            0x0E927AED,
            since = 150
        ) { _sceKernelReturnFromTimerHandler(it) }
        registerFunctionRaw("sceKernelUSec2SysClock", 0x110DEC9A, since = 150) { sceKernelUSec2SysClock(it) }
        registerFunctionRaw("sceKernelCreateVTimer", 0x20FFF560, since = 150) { sceKernelCreateVTimer(it) }
        registerFunctionRaw("sceKernelGetCallbackCount", 0x2A3D44FF, since = 150) { sceKernelGetCallbackCount(it) }
        registerFunctionRaw("ThreadManForUser_31327F19", 0x31327F19, since = 150) { ThreadManForUser_31327F19(it) }
        registerFunctionRaw("sceKernelDeleteVTimer", 0x328F9E52, since = 150) { sceKernelDeleteVTimer(it) }
        registerFunctionRaw("sceKernelReferMsgPipeStatus", 0x33BE4024, since = 150) { sceKernelReferMsgPipeStatus(it) }
        registerFunctionRaw("sceKernelCancelMsgPipe", 0x349B864D, since = 150) { sceKernelCancelMsgPipe(it) }
        registerFunctionRaw(
            "sceKernelSetVTimerHandlerWide",
            0x53B00E9A,
            since = 150
        ) { sceKernelSetVTimerHandlerWide(it) }
        registerFunctionRaw("sceKernelSetVTimerTime", 0x542AD630, since = 150) { sceKernelSetVTimerTime(it) }
        registerFunctionRaw("sceKernelReferVTimerStatus", 0x5F32BEAA, since = 150) { sceKernelReferVTimerStatus(it) }
        registerFunctionRaw("sceKernelReferSystemStatus", 0x627E6F3A, since = 150) { sceKernelReferSystemStatus(it) }
        registerFunctionRaw(
            "_sceKernelReturnFromCallback",
            0x6E9EA350,
            since = 150
        ) { _sceKernelReturnFromCallback(it) }
        registerFunctionRaw("ThreadManForUser_71040D5C", 0x71040D5C, since = 150) { ThreadManForUser_71040D5C(it) }
        registerFunctionRaw(
            "sceKernelReleaseThreadEventHandler",
            0x72F3C145,
            since = 150
        ) { sceKernelReleaseThreadEventHandler(it) }
        registerFunctionRaw(
            "sceKernelReferCallbackStatus",
            0x730ED8BC,
            since = 150
        ) { sceKernelReferCallbackStatus(it) }
        registerFunctionRaw("sceKernelReceiveMsgPipe", 0x74829B76, since = 150) { sceKernelReceiveMsgPipe(it) }
        registerFunctionRaw("sceKernelCreateMsgPipe", 0x7C0DC2A0, since = 150) { sceKernelCreateMsgPipe(it) }
        registerFunctionRaw("sceKernelSendMsgPipeCB", 0x7C41F2C2, since = 150) { sceKernelSendMsgPipeCB(it) }
        registerFunctionRaw("ThreadManForUser_7CFF8CF3", 0x7CFF8CF3, since = 150) { ThreadManForUser_7CFF8CF3(it) }
        registerFunctionRaw(
            "sceKernelReferGlobalProfiler",
            0x8218B4DD,
            since = 150
        ) { sceKernelReferGlobalProfiler(it) }
        registerFunctionRaw("ThreadManForUser_8672E3D0", 0x8672E3D0, since = 150) { ThreadManForUser_8672E3D0(it) }
        registerFunctionRaw("sceKernelSendMsgPipe", 0x876DBFAD, since = 150) { sceKernelSendMsgPipe(it) }
        registerFunctionRaw("sceKernelTrySendMsgPipe", 0x884C9F90, since = 150) { sceKernelTrySendMsgPipe(it) }
        registerFunctionRaw("sceKernelGetVTimerBase", 0xB3A59970, since = 150) { sceKernelGetVTimerBase(it) }
        registerFunctionRaw("sceKernelGetVTimerBaseWide", 0xB7C18B77, since = 150) { sceKernelGetVTimerBaseWide(it) }
        registerFunctionRaw("sceKernelSysClock2USec", 0xBA6B92E2, since = 150) { sceKernelSysClock2USec(it) }
        registerFunctionRaw("ThreadManForUser_BEED3A47", 0xBEED3A47, since = 150) { ThreadManForUser_BEED3A47(it) }
        registerFunctionRaw("sceKernelGetVTimerTimeWide", 0xC0B3FFD2, since = 150) { sceKernelGetVTimerTimeWide(it) }
        registerFunctionRaw("sceKernelStartVTimer", 0xC68D9437, since = 150) { sceKernelStartVTimer(it) }
        registerFunctionRaw("sceKernelUSec2SysClockWide", 0xC8CD158C, since = 150) { sceKernelUSec2SysClockWide(it) }
        registerFunctionRaw("sceKernelStopVTimer", 0xD0AEEE87, since = 150) { sceKernelStopVTimer(it) }
        registerFunctionRaw(
            "sceKernelCancelVTimerHandler",
            0xD2D615EF,
            since = 150
        ) { sceKernelCancelVTimerHandler(it) }
        registerFunctionRaw("sceKernelSetVTimerHandler", 0xD8B299AE, since = 150) { sceKernelSetVTimerHandler(it) }
        registerFunctionRaw("sceKernelTryReceiveMsgPipe", 0xDF52098F, since = 150) { sceKernelTryReceiveMsgPipe(it) }
        registerFunctionRaw("sceKernelSysClock2USecWide", 0xE1619D7C, since = 150) { sceKernelSysClock2USecWide(it) }
        registerFunctionRaw("sceKernelDeleteMsgPipe", 0xF0B7DA1C, since = 150) { sceKernelDeleteMsgPipe(it) }
        registerFunctionRaw("sceKernelSetVTimerTimeWide", 0xFB6425C3, since = 150) { sceKernelSetVTimerTimeWide(it) }
        registerFunctionRaw("sceKernelReceiveMsgPipeCB", 0xFBFA697D, since = 150) { sceKernelReceiveMsgPipeCB(it) }
    }
}

class SceKernelSemaInfo(
    var size: Int = 0,
    var name: String = "",
    var attributes: Int = 0, // SemaphoreAttribute
    var initialCount: Int = 0,
    var currentCount: Int = 0,
    var maximumCount: Int = 0,
    var numberOfWaitingThreads: Int = 0
) {
    companion object : Struct<SceKernelSemaInfo>(
        { SceKernelSemaInfo() },
        SceKernelSemaInfo::size AS INT32,
        SceKernelSemaInfo::name AS STRINGZ(32),
        SceKernelSemaInfo::attributes AS INT32,
        SceKernelSemaInfo::initialCount AS INT32,
        SceKernelSemaInfo::currentCount AS INT32,
        SceKernelSemaInfo::maximumCount AS INT32,
        SceKernelSemaInfo::numberOfWaitingThreads AS INT32
    )
}

class K0Structure(
    var unk: IntArray = IntArray(48), // +0000
    var threadId: Int = 0, // +00C0
    var unk1: Int = 0, // +00C4
    var stackAddr: Int = 0, // +00C8
    var unk3: IntArray = IntArray(11),// +00CC
    var f1: Int = 0, // +00F8
    var f2: Int = 0 // +00FC
) {
    companion object : Struct<K0Structure>(
        { K0Structure() },
        K0Structure::unk AS INTLIKEARRAY(INT32, 48),
        K0Structure::threadId AS INT32,
        K0Structure::unk1 AS INT32,
        K0Structure::stackAddr AS INT32,
        K0Structure::unk3 AS INTLIKEARRAY(INT32, 11),
        K0Structure::f1 AS INT32,
        K0Structure::f2 AS INT32
    )
}