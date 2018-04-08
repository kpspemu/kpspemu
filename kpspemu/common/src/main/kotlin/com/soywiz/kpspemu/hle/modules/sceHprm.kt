package com.soywiz.kpspemu.hle.modules

import com.soywiz.kpspemu.*
import com.soywiz.kpspemu.cpu.*
import com.soywiz.kpspemu.hle.*
import com.soywiz.kpspemu.mem.*

class sceHprm(emulator: Emulator) :
    SceModule(emulator, "sceHprm", 0x40010011, "hpremote_02g.prx", "sceHP_Remote_Driver") {
    fun sceHprmPeekCurrentKey(ptr: Ptr): Int {
        ptr.sw(0, 0)
        return 0
    }

    fun sceHprmIsRemoteExist(cpu: CpuState): Unit = UNIMPLEMENTED(0x208DB1BD)
    fun sceHprmIsMicrophoneExist(cpu: CpuState): Unit = UNIMPLEMENTED(0x219C58F1)
    fun sceHprmPeekLatch(cpu: CpuState): Unit = UNIMPLEMENTED(0x2BCEC83E)
    fun sceHprm_3953DE6B(cpu: CpuState): Unit = UNIMPLEMENTED(0x3953DE6B)
    fun sceHprm_396FD885(cpu: CpuState): Unit = UNIMPLEMENTED(0x396FD885)
    fun sceHprmReadLatch(cpu: CpuState): Unit = UNIMPLEMENTED(0x40D2F9F0)
    fun sceHprmUnregitserCallback(cpu: CpuState): Unit = UNIMPLEMENTED(0x444ED0B7)
    fun sceHprmGetHpDetect(cpu: CpuState): Unit = UNIMPLEMENTED(0x71B5FB67)
    fun sceHprmIsHeadphoneExist(cpu: CpuState): Unit = UNIMPLEMENTED(0x7E69EDA4)
    fun sceHprmRegisterCallback(cpu: CpuState): Unit = UNIMPLEMENTED(0xC7154136)
    fun sceHprm_FD7DE6CD(cpu: CpuState): Unit = UNIMPLEMENTED(0xFD7DE6CD)


    override fun registerModule() {
        registerFunctionInt("sceHprmPeekCurrentKey", 0x1910B327, since = 150) { sceHprmPeekCurrentKey(ptr) }

        registerFunctionRaw("sceHprmIsRemoteExist", 0x208DB1BD, since = 150) { sceHprmIsRemoteExist(it) }
        registerFunctionRaw("sceHprmIsMicrophoneExist", 0x219C58F1, since = 150) { sceHprmIsMicrophoneExist(it) }
        registerFunctionRaw("sceHprmPeekLatch", 0x2BCEC83E, since = 150) { sceHprmPeekLatch(it) }
        registerFunctionRaw("sceHprm_3953DE6B", 0x3953DE6B, since = 150) { sceHprm_3953DE6B(it) }
        registerFunctionRaw("sceHprm_396FD885", 0x396FD885, since = 150) { sceHprm_396FD885(it) }
        registerFunctionRaw("sceHprmReadLatch", 0x40D2F9F0, since = 150) { sceHprmReadLatch(it) }
        registerFunctionRaw("sceHprmUnregitserCallback", 0x444ED0B7, since = 150) { sceHprmUnregitserCallback(it) }
        registerFunctionRaw("sceHprmGetHpDetect", 0x71B5FB67, since = 150) { sceHprmGetHpDetect(it) }
        registerFunctionRaw("sceHprmIsHeadphoneExist", 0x7E69EDA4, since = 150) { sceHprmIsHeadphoneExist(it) }
        registerFunctionRaw("sceHprmRegisterCallback", 0xC7154136, since = 150) { sceHprmRegisterCallback(it) }
        registerFunctionRaw("sceHprm_FD7DE6CD", 0xFD7DE6CD, since = 150) { sceHprm_FD7DE6CD(it) }
    }

    companion object {
        const val PSP_HPRM_PLAYPAUSE = 0x1
        const val PSP_HPRM_FORWARD = 0x4
        const val PSP_HPRM_BACK = 0x8
        const val PSP_HPRM_VOL_UP = 0x10
        const val PSP_HPRM_VOL_DOWN = 0x20
        const val PSP_HPRM_HOLD = 0x8
    }
}
