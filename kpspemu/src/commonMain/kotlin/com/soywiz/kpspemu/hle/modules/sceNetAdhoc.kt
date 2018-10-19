package com.soywiz.kpspemu.hle.modules

import com.soywiz.kpspemu.*
import com.soywiz.kpspemu.cpu.*
import com.soywiz.kpspemu.hle.*

class sceNetAdhoc(emulator: Emulator) :
    SceModule(emulator, "sceNetAdhoc", 0x00010011, "pspnet_adhoc.prx", "sceNetAdhoc_Library") {
    fun sceNetAdhocGameModeDeleteReplica(cpu: CpuState): Unit = UNIMPLEMENTED(0x0B2228E9)
    fun sceNetAdhocPtpClose(cpu: CpuState): Unit = UNIMPLEMENTED(0x157E6225)
    fun sceNetAdhocGameModeCreateReplica(cpu: CpuState): Unit = UNIMPLEMENTED(0x3278AB0C)
    fun sceNetAdhocGetSocketAlert(cpu: CpuState): Unit = UNIMPLEMENTED(0x4D2CE199)
    fun sceNetAdhocPtpSend(cpu: CpuState): Unit = UNIMPLEMENTED(0x4DA4C788)
    fun sceNetAdhoc_67346A2A(cpu: CpuState): Unit = UNIMPLEMENTED(0x67346A2A)
    fun sceNetAdhocPdpCreate(cpu: CpuState): Unit = UNIMPLEMENTED(0x6F92741B)
    fun sceNetAdhocSetSocketAlert(cpu: CpuState): Unit = UNIMPLEMENTED(0x73BFD52D)
    fun sceNetAdhocPollSocket(cpu: CpuState): Unit = UNIMPLEMENTED(0x7A662D6B)
    fun sceNetAdhocPdpDelete(cpu: CpuState): Unit = UNIMPLEMENTED(0x7F27BB5E)
    fun sceNetAdhocGameModeCreateMaster(cpu: CpuState): Unit = UNIMPLEMENTED(0x7F75C338)
    fun sceNetAdhocPtpOpen(cpu: CpuState): Unit = UNIMPLEMENTED(0x877F6D66)
    fun sceNetAdhocPtpRecv(cpu: CpuState): Unit = UNIMPLEMENTED(0x8BEA2B3E)
    fun sceNetAdhocGameModeUpdateMaster(cpu: CpuState): Unit = UNIMPLEMENTED(0x98C204C8)
    fun sceNetAdhocPtpFlush(cpu: CpuState): Unit = UNIMPLEMENTED(0x9AC2EEAC)
    fun sceNetAdhocPtpAccept(cpu: CpuState): Unit = UNIMPLEMENTED(0x9DF81198)
    fun sceNetAdhocGameModeDeleteMaster(cpu: CpuState): Unit = UNIMPLEMENTED(0xA0229362)
    fun sceNetAdhocTerm(cpu: CpuState): Unit = UNIMPLEMENTED(0xA62C6F57)
    fun sceNetAdhocPdpSend(cpu: CpuState): Unit = UNIMPLEMENTED(0xABED3790)
    fun sceNetAdhocGetPtpStat(cpu: CpuState): Unit = UNIMPLEMENTED(0xB9685118)
    fun sceNetAdhocGetPdpStat(cpu: CpuState): Unit = UNIMPLEMENTED(0xC7C1FC57)
    fun sceNetAdhocPdpRecv(cpu: CpuState): Unit = UNIMPLEMENTED(0xDFE53E03)
    fun sceNetAdhocPtpListen(cpu: CpuState): Unit = UNIMPLEMENTED(0xE08BDAC1)
    fun sceNetAdhocInit(cpu: CpuState): Unit = UNIMPLEMENTED(0xE1D621D7)
    fun sceNetAdhocGameModeUpdateReplica(cpu: CpuState): Unit = UNIMPLEMENTED(0xFA324B4E)
    fun sceNetAdhocPtpConnect(cpu: CpuState): Unit = UNIMPLEMENTED(0xFC6FC07B)


    override fun registerModule() {
        registerFunctionRaw(
            "sceNetAdhocGameModeDeleteReplica",
            0x0B2228E9,
            since = 150
        ) { sceNetAdhocGameModeDeleteReplica(it) }
        registerFunctionRaw("sceNetAdhocPtpClose", 0x157E6225, since = 150) { sceNetAdhocPtpClose(it) }
        registerFunctionRaw(
            "sceNetAdhocGameModeCreateReplica",
            0x3278AB0C,
            since = 150
        ) { sceNetAdhocGameModeCreateReplica(it) }
        registerFunctionRaw("sceNetAdhocGetSocketAlert", 0x4D2CE199, since = 150) { sceNetAdhocGetSocketAlert(it) }
        registerFunctionRaw("sceNetAdhocPtpSend", 0x4DA4C788, since = 150) { sceNetAdhocPtpSend(it) }
        registerFunctionRaw("sceNetAdhoc_67346A2A", 0x67346A2A, since = 150) { sceNetAdhoc_67346A2A(it) }
        registerFunctionRaw("sceNetAdhocPdpCreate", 0x6F92741B, since = 150) { sceNetAdhocPdpCreate(it) }
        registerFunctionRaw("sceNetAdhocSetSocketAlert", 0x73BFD52D, since = 150) { sceNetAdhocSetSocketAlert(it) }
        registerFunctionRaw("sceNetAdhocPollSocket", 0x7A662D6B, since = 150) { sceNetAdhocPollSocket(it) }
        registerFunctionRaw("sceNetAdhocPdpDelete", 0x7F27BB5E, since = 150) { sceNetAdhocPdpDelete(it) }
        registerFunctionRaw(
            "sceNetAdhocGameModeCreateMaster",
            0x7F75C338,
            since = 150
        ) { sceNetAdhocGameModeCreateMaster(it) }
        registerFunctionRaw("sceNetAdhocPtpOpen", 0x877F6D66, since = 150) { sceNetAdhocPtpOpen(it) }
        registerFunctionRaw("sceNetAdhocPtpRecv", 0x8BEA2B3E, since = 150) { sceNetAdhocPtpRecv(it) }
        registerFunctionRaw(
            "sceNetAdhocGameModeUpdateMaster",
            0x98C204C8,
            since = 150
        ) { sceNetAdhocGameModeUpdateMaster(it) }
        registerFunctionRaw("sceNetAdhocPtpFlush", 0x9AC2EEAC, since = 150) { sceNetAdhocPtpFlush(it) }
        registerFunctionRaw("sceNetAdhocPtpAccept", 0x9DF81198, since = 150) { sceNetAdhocPtpAccept(it) }
        registerFunctionRaw(
            "sceNetAdhocGameModeDeleteMaster",
            0xA0229362,
            since = 150
        ) { sceNetAdhocGameModeDeleteMaster(it) }
        registerFunctionRaw("sceNetAdhocTerm", 0xA62C6F57, since = 150) { sceNetAdhocTerm(it) }
        registerFunctionRaw("sceNetAdhocPdpSend", 0xABED3790, since = 150) { sceNetAdhocPdpSend(it) }
        registerFunctionRaw("sceNetAdhocGetPtpStat", 0xB9685118, since = 150) { sceNetAdhocGetPtpStat(it) }
        registerFunctionRaw("sceNetAdhocGetPdpStat", 0xC7C1FC57, since = 150) { sceNetAdhocGetPdpStat(it) }
        registerFunctionRaw("sceNetAdhocPdpRecv", 0xDFE53E03, since = 150) { sceNetAdhocPdpRecv(it) }
        registerFunctionRaw("sceNetAdhocPtpListen", 0xE08BDAC1, since = 150) { sceNetAdhocPtpListen(it) }
        registerFunctionRaw("sceNetAdhocInit", 0xE1D621D7, since = 150) { sceNetAdhocInit(it) }
        registerFunctionRaw(
            "sceNetAdhocGameModeUpdateReplica",
            0xFA324B4E,
            since = 150
        ) { sceNetAdhocGameModeUpdateReplica(it) }
        registerFunctionRaw("sceNetAdhocPtpConnect", 0xFC6FC07B, since = 150) { sceNetAdhocPtpConnect(it) }
    }
}
