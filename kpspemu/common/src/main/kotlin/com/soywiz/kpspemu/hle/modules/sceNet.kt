package com.soywiz.kpspemu.hle.modules

import com.soywiz.kpspemu.*
import com.soywiz.kpspemu.cpu.*
import com.soywiz.kpspemu.hle.*

class sceNet(emulator: Emulator) : SceModule(emulator, "sceNet", 0x00010011, "pspnet.prx", "sceNet_Library") {
    fun sceNetGetLocalEtherAddr(cpu: CpuState): Unit = UNIMPLEMENTED(0x0BF0A3AE)
    fun sceNetTerm(cpu: CpuState): Unit = UNIMPLEMENTED(0x281928A9)
    fun sceNetInit(cpu: CpuState): Unit = UNIMPLEMENTED(0x39AF39A6)
    fun sceNet_3B617AA0(cpu: CpuState): Unit = UNIMPLEMENTED(0x3B617AA0)
    fun sceNetFreeThreadinfo(cpu: CpuState): Unit = UNIMPLEMENTED(0x50647530)
    fun sceNetEtherNtostr(cpu: CpuState): Unit = UNIMPLEMENTED(0x89360950)
    fun sceNetThreadAbort(cpu: CpuState): Unit = UNIMPLEMENTED(0xAD6844C6)
    fun sceNet_B6FC0A5B(cpu: CpuState): Unit = UNIMPLEMENTED(0xB6FC0A5B)
    fun sceNet_BFCFEFF6(cpu: CpuState): Unit = UNIMPLEMENTED(0xBFCFEFF6)
    fun sceNet_C431A214(cpu: CpuState): Unit = UNIMPLEMENTED(0xC431A214)
    fun sceNetGetMallocStat(cpu: CpuState): Unit = UNIMPLEMENTED(0xCC393E48)
    fun sceNetEtherStrton(cpu: CpuState): Unit = UNIMPLEMENTED(0xD27961C9)
    fun sceNet_DB88F458(cpu: CpuState): Unit = UNIMPLEMENTED(0xDB88F458)
    fun sceNet_E1F4696F(cpu: CpuState): Unit = UNIMPLEMENTED(0xE1F4696F)

    override fun registerModule() {
        registerFunctionRaw("sceNetGetLocalEtherAddr", 0x0BF0A3AE, since = 150) { sceNetGetLocalEtherAddr(it) }
        registerFunctionRaw("sceNetTerm", 0x281928A9, since = 150) { sceNetTerm(it) }
        registerFunctionRaw("sceNetInit", 0x39AF39A6, since = 150) { sceNetInit(it) }
        registerFunctionRaw("sceNet_3B617AA0", 0x3B617AA0, since = 150) { sceNet_3B617AA0(it) }
        registerFunctionRaw("sceNetFreeThreadinfo", 0x50647530, since = 150) { sceNetFreeThreadinfo(it) }
        registerFunctionRaw("sceNetEtherNtostr", 0x89360950, since = 150) { sceNetEtherNtostr(it) }
        registerFunctionRaw("sceNetThreadAbort", 0xAD6844C6, since = 150) { sceNetThreadAbort(it) }
        registerFunctionRaw("sceNet_B6FC0A5B", 0xB6FC0A5B, since = 150) { sceNet_B6FC0A5B(it) }
        registerFunctionRaw("sceNet_BFCFEFF6", 0xBFCFEFF6, since = 150) { sceNet_BFCFEFF6(it) }
        registerFunctionRaw("sceNet_C431A214", 0xC431A214, since = 150) { sceNet_C431A214(it) }
        registerFunctionRaw("sceNetGetMallocStat", 0xCC393E48, since = 150) { sceNetGetMallocStat(it) }
        registerFunctionRaw("sceNetEtherStrton", 0xD27961C9, since = 150) { sceNetEtherStrton(it) }
        registerFunctionRaw("sceNet_DB88F458", 0xDB88F458, since = 150) { sceNet_DB88F458(it) }
        registerFunctionRaw("sceNet_E1F4696F", 0xE1F4696F, since = 150) { sceNet_E1F4696F(it) }
    }
}
