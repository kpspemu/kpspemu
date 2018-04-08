package com.soywiz.kpspemu.hle.modules

import com.soywiz.kpspemu.*
import com.soywiz.kpspemu.cpu.*
import com.soywiz.kpspemu.hle.*

class sceNetAdhocctl(emulator: Emulator) :
    SceModule(emulator, "sceNetAdhocctl", 0x00010011, "pspnet_adhocctl.prx", "sceNetAdhocctl_Library") {
    fun sceNetAdhocctlScan(cpu: CpuState): Unit = UNIMPLEMENTED(0x08FFF7A0)
    fun sceNetAdhocctlConnect(cpu: CpuState): Unit = UNIMPLEMENTED(0x0AD043ED)
    fun sceNetAdhocctlJoinEnterGameMode(cpu: CpuState): Unit = UNIMPLEMENTED(0x1FF89745)
    fun sceNetAdhocctlAddHandler(cpu: CpuState): Unit = UNIMPLEMENTED(0x20B317A0)
    fun sceNetAdhocctlDisconnect(cpu: CpuState): Unit = UNIMPLEMENTED(0x34401D65)
    fun sceNetAdhocctlGetAdhocId(cpu: CpuState): Unit = UNIMPLEMENTED(0x362CBE8F)
    fun sceNetAdhocctlGetGameModeInfo(cpu: CpuState): Unit = UNIMPLEMENTED(0x5A014CE0)
    fun sceNetAdhocctlJoin(cpu: CpuState): Unit = UNIMPLEMENTED(0x5E7F79C9)
    fun sceNetAdhocctlDelHandler(cpu: CpuState): Unit = UNIMPLEMENTED(0x6402490B)
    fun sceNetAdhocctlGetState(cpu: CpuState): Unit = UNIMPLEMENTED(0x75ECD386)
    fun sceNetAdhocctlGetScanInfo(cpu: CpuState): Unit = UNIMPLEMENTED(0x81AEE1BE)
    fun sceNetAdhocctlGetNameByAddr(cpu: CpuState): Unit = UNIMPLEMENTED(0x8916C003)
    fun sceNetAdhocctlGetPeerInfo(cpu: CpuState): Unit = UNIMPLEMENTED(0x8DB83FDC)
    fun sceNetAdhocctlGetAddrByName(cpu: CpuState): Unit = UNIMPLEMENTED(0x99560ABE)
    fun sceNetAdhocctlTerm(cpu: CpuState): Unit = UNIMPLEMENTED(0x9D689E13)
    fun sceNetAdhocctlCreateEnterGameMode(cpu: CpuState): Unit = UNIMPLEMENTED(0xA5C055CE)
    fun sceNetAdhocctl_B0B80E80(cpu: CpuState): Unit = UNIMPLEMENTED(0xB0B80E80)
    fun sceNetAdhocctlExitGameMode(cpu: CpuState): Unit = UNIMPLEMENTED(0xCF8E084D)
    fun sceNetAdhocctlGetParameter(cpu: CpuState): Unit = UNIMPLEMENTED(0xDED9D28E)
    fun sceNetAdhocctlGetPeerList(cpu: CpuState): Unit = UNIMPLEMENTED(0xE162CB14)
    fun sceNetAdhocctlInit(cpu: CpuState): Unit = UNIMPLEMENTED(0xE26F226E)
    fun sceNetAdhocctlCreate(cpu: CpuState): Unit = UNIMPLEMENTED(0xEC0635C1)


    override fun registerModule() {
        registerFunctionRaw("sceNetAdhocctlScan", 0x08FFF7A0, since = 150) { sceNetAdhocctlScan(it) }
        registerFunctionRaw("sceNetAdhocctlConnect", 0x0AD043ED, since = 150) { sceNetAdhocctlConnect(it) }
        registerFunctionRaw(
            "sceNetAdhocctlJoinEnterGameMode",
            0x1FF89745,
            since = 150
        ) { sceNetAdhocctlJoinEnterGameMode(it) }
        registerFunctionRaw("sceNetAdhocctlAddHandler", 0x20B317A0, since = 150) { sceNetAdhocctlAddHandler(it) }
        registerFunctionRaw("sceNetAdhocctlDisconnect", 0x34401D65, since = 150) { sceNetAdhocctlDisconnect(it) }
        registerFunctionRaw("sceNetAdhocctlGetAdhocId", 0x362CBE8F, since = 150) { sceNetAdhocctlGetAdhocId(it) }
        registerFunctionRaw(
            "sceNetAdhocctlGetGameModeInfo",
            0x5A014CE0,
            since = 150
        ) { sceNetAdhocctlGetGameModeInfo(it) }
        registerFunctionRaw("sceNetAdhocctlJoin", 0x5E7F79C9, since = 150) { sceNetAdhocctlJoin(it) }
        registerFunctionRaw("sceNetAdhocctlDelHandler", 0x6402490B, since = 150) { sceNetAdhocctlDelHandler(it) }
        registerFunctionRaw("sceNetAdhocctlGetState", 0x75ECD386, since = 150) { sceNetAdhocctlGetState(it) }
        registerFunctionRaw("sceNetAdhocctlGetScanInfo", 0x81AEE1BE, since = 150) { sceNetAdhocctlGetScanInfo(it) }
        registerFunctionRaw("sceNetAdhocctlGetNameByAddr", 0x8916C003, since = 150) { sceNetAdhocctlGetNameByAddr(it) }
        registerFunctionRaw("sceNetAdhocctlGetPeerInfo", 0x8DB83FDC, since = 150) { sceNetAdhocctlGetPeerInfo(it) }
        registerFunctionRaw("sceNetAdhocctlGetAddrByName", 0x99560ABE, since = 150) { sceNetAdhocctlGetAddrByName(it) }
        registerFunctionRaw("sceNetAdhocctlTerm", 0x9D689E13, since = 150) { sceNetAdhocctlTerm(it) }
        registerFunctionRaw(
            "sceNetAdhocctlCreateEnterGameMode",
            0xA5C055CE,
            since = 150
        ) { sceNetAdhocctlCreateEnterGameMode(it) }
        registerFunctionRaw("sceNetAdhocctl_B0B80E80", 0xB0B80E80, since = 150) { sceNetAdhocctl_B0B80E80(it) }
        registerFunctionRaw("sceNetAdhocctlExitGameMode", 0xCF8E084D, since = 150) { sceNetAdhocctlExitGameMode(it) }
        registerFunctionRaw("sceNetAdhocctlGetParameter", 0xDED9D28E, since = 150) { sceNetAdhocctlGetParameter(it) }
        registerFunctionRaw("sceNetAdhocctlGetPeerList", 0xE162CB14, since = 150) { sceNetAdhocctlGetPeerList(it) }
        registerFunctionRaw("sceNetAdhocctlInit", 0xE26F226E, since = 150) { sceNetAdhocctlInit(it) }
        registerFunctionRaw("sceNetAdhocctlCreate", 0xEC0635C1, since = 150) { sceNetAdhocctlCreate(it) }
    }
}
