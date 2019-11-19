package com.soywiz.kpspemu.hle

import com.soywiz.kpspemu.*
import com.soywiz.kpspemu.hle.manager.*
import com.soywiz.kpspemu.hle.modules.*
import com.soywiz.kpspemu.hle.modules.InterruptManager

fun Emulator.registerNativeModules() = moduleManager.registerNativeModules()

fun ModuleManager.registerNativeModules() = this.apply {
    register(LoadExecForUser(emulator))
    register(ThreadManForUser(emulator))
    register(SysMemUserForUser(emulator))
    register(StdioForUser(emulator))
    register(ModuleMgrForUser(emulator))
    register(IoFileMgrForUser(emulator))
    register(sceGe_user(emulator))
    register(sceRtc(emulator))
    register(sceCtrl(emulator))
    register(sceDisplay(emulator))
    register(UtilsForKernel(emulator))
    register(UtilsForUser(emulator))
    register(scePower(emulator))
    register(sceNetInet(emulator))
    register(sceAudio(emulator))
    register(sceSasCore(emulator))
    register(sceMpeg(emulator))
    register(sceAtrac3plus(emulator))
    register(sceUmdUser(emulator))
    register(sceUtility(emulator))
    register(sceImpose(emulator))
    register(InterruptManager(emulator))
    register(Kernel_Library(emulator))
    register(sceSuspendForUser(emulator))
    register(sceDmac(emulator))
    register(ExceptionManagerForKernel(emulator))
    register(LoadCoreForKernel(emulator))
    register(sceHprm(emulator))
    register(sceNet(emulator))
    register(sceNetAdhoc(emulator))
    register(sceNetAdhocctl(emulator))
    register(sceNetAdhocMatching(emulator))
    register(sceWlanDrv(emulator))
    register(sceReg(emulator))
    register(sceMp3(emulator))
    register(sceVaudio(emulator))
    register(sceLibFont(emulator))
    register(pspDveManager(emulator))
}