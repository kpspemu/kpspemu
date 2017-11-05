package com.soywiz.kpspemu.hle

import com.soywiz.kpspemu.Emulator
import com.soywiz.kpspemu.hle.manager.ModuleManager
import com.soywiz.kpspemu.hle.modules.*

fun Emulator.registerNativeModules() = moduleManager.registerNativeModules()

fun ModuleManager.registerNativeModules() {
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
}