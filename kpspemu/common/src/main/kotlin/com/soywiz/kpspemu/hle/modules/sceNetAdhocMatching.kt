package com.soywiz.kpspemu.hle.modules

import com.soywiz.kpspemu.*
import com.soywiz.kpspemu.cpu.*
import com.soywiz.kpspemu.hle.*

class sceNetAdhocMatching(emulator: Emulator) :
    SceModule(emulator, "sceNetAdhocMatching", 0x00010011, "pspnet_adhoc_matching.prx", "sceNetAdhocMatching_Library") {
    fun sceNetAdhocMatchingInit(cpu: CpuState): Unit = UNIMPLEMENTED(0x2A2A1E07)
    fun sceNetAdhocMatchingStop(cpu: CpuState): Unit = UNIMPLEMENTED(0x32B156B3)
    fun sceNetAdhocMatchingGetPoolMaxAlloc(cpu: CpuState): Unit = UNIMPLEMENTED(0x40F8F435)
    fun sceNetAdhocMatchingSelectTarget(cpu: CpuState): Unit = UNIMPLEMENTED(0x5E3D4B79)
    fun sceNetAdhocMatchingTerm(cpu: CpuState): Unit = UNIMPLEMENTED(0x7945ECDA)
    fun sceNetAdhocMatchingCancelTargetWithOpt(cpu: CpuState): Unit = UNIMPLEMENTED(0x8F58BEDF)
    fun sceNetAdhocMatchingStart(cpu: CpuState): Unit = UNIMPLEMENTED(0x93EF3843)
    fun sceNetAdhocMatchingGetPoolStat(cpu: CpuState): Unit = UNIMPLEMENTED(0x9C5CFB7D)
    fun sceNetAdhocMatchingSetHelloOpt(cpu: CpuState): Unit = UNIMPLEMENTED(0xB58E61B7)
    fun sceNetAdhocMatchingGetHelloOpt(cpu: CpuState): Unit = UNIMPLEMENTED(0xB5D96C2A)
    fun sceNetAdhocMatchingGetMembers(cpu: CpuState): Unit = UNIMPLEMENTED(0xC58BCD9E)
    fun sceNetAdhocMatchingCreate(cpu: CpuState): Unit = UNIMPLEMENTED(0xCA5EDA6F)
    fun sceNetAdhocMatchingCancelTarget(cpu: CpuState): Unit = UNIMPLEMENTED(0xEA3C6108)
    fun sceNetAdhocMatchingAbortSendData(cpu: CpuState): Unit = UNIMPLEMENTED(0xEC19337D)
    fun sceNetAdhocMatchingDelete(cpu: CpuState): Unit = UNIMPLEMENTED(0xF16EAF4F)
    fun sceNetAdhocMatchingSendData(cpu: CpuState): Unit = UNIMPLEMENTED(0xF79472D7)

    override fun registerModule() {
        registerFunctionRaw("sceNetAdhocMatchingInit", 0x2A2A1E07, since = 150) { sceNetAdhocMatchingInit(it) }
        registerFunctionRaw("sceNetAdhocMatchingStop", 0x32B156B3, since = 150) { sceNetAdhocMatchingStop(it) }
        registerFunctionRaw(
            "sceNetAdhocMatchingGetPoolMaxAlloc",
            0x40F8F435,
            since = 150
        ) { sceNetAdhocMatchingGetPoolMaxAlloc(it) }
        registerFunctionRaw(
            "sceNetAdhocMatchingSelectTarget",
            0x5E3D4B79,
            since = 150
        ) { sceNetAdhocMatchingSelectTarget(it) }
        registerFunctionRaw("sceNetAdhocMatchingTerm", 0x7945ECDA, since = 150) { sceNetAdhocMatchingTerm(it) }
        registerFunctionRaw(
            "sceNetAdhocMatchingCancelTargetWithOpt",
            0x8F58BEDF,
            since = 150
        ) { sceNetAdhocMatchingCancelTargetWithOpt(it) }
        registerFunctionRaw("sceNetAdhocMatchingStart", 0x93EF3843, since = 150) { sceNetAdhocMatchingStart(it) }
        registerFunctionRaw("sceNetAdhocMatchingGetPoolStat", 0x9C5CFB7D, since = 150) {
            sceNetAdhocMatchingGetPoolStat(
                it
            )
        }
        registerFunctionRaw("sceNetAdhocMatchingSetHelloOpt", 0xB58E61B7, since = 150) {
            sceNetAdhocMatchingSetHelloOpt(
                it
            )
        }
        registerFunctionRaw("sceNetAdhocMatchingGetHelloOpt", 0xB5D96C2A, since = 150) {
            sceNetAdhocMatchingGetHelloOpt(
                it
            )
        }
        registerFunctionRaw(
            "sceNetAdhocMatchingGetMembers",
            0xC58BCD9E,
            since = 150
        ) { sceNetAdhocMatchingGetMembers(it) }
        registerFunctionRaw("sceNetAdhocMatchingCreate", 0xCA5EDA6F, since = 150) { sceNetAdhocMatchingCreate(it) }
        registerFunctionRaw(
            "sceNetAdhocMatchingCancelTarget",
            0xEA3C6108,
            since = 150
        ) { sceNetAdhocMatchingCancelTarget(it) }
        registerFunctionRaw(
            "sceNetAdhocMatchingAbortSendData",
            0xEC19337D,
            since = 150
        ) { sceNetAdhocMatchingAbortSendData(it) }
        registerFunctionRaw("sceNetAdhocMatchingDelete", 0xF16EAF4F, since = 150) { sceNetAdhocMatchingDelete(it) }
        registerFunctionRaw("sceNetAdhocMatchingSendData", 0xF79472D7, since = 150) { sceNetAdhocMatchingSendData(it) }
    }
}
