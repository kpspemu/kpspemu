package com.soywiz.kpspemu.hle.modules

import com.soywiz.kpspemu.*
import com.soywiz.kpspemu.cpu.*
import com.soywiz.kpspemu.cpu.dis.*
import com.soywiz.kpspemu.hle.*
import com.soywiz.kpspemu.hle.manager.*

class LoadExecForUser(emulator: Emulator) :
    SceModule(emulator, "LoadExecForUser", 0x40010011, "loadexec_02g.prx", "sceLoadExec") {
    fun sceKernelExitGame(currentThread: PspThread): Unit {
        logger.info { "sceKernelExitGame: '${currentThread.name}'" }
        logger.info { mem.getPrintInstructionAt(currentThread.state.PC, emulator.nameProvider) }
        logger.info { mem.getPrintInstructionAt(currentThread.state.RA, emulator.nameProvider) }
        threadManager.stopAllThreads()
    }

    fun sceKernelExitGameWithStatus(cpu: CpuState): Unit = UNIMPLEMENTED(0x2AC9954B)
    fun LoadExecForUser_362A956B(cpu: CpuState): Unit = UNIMPLEMENTED(0x362A956B)

    fun sceKernelRegisterExitCallback(callbackId: Int): Int {
        println("Unimplemented: sceKernelRegisterExitCallback: $callbackId")
        return 0
    }

    fun LoadExecForUser_8ADA38D3(cpu: CpuState): Unit = UNIMPLEMENTED(0x8ADA38D3)
    fun sceKernelLoadExec(cpu: CpuState): Unit = UNIMPLEMENTED(0xBD2F1094)
    fun LoadExecForUser_D1FB50DC(cpu: CpuState): Unit = UNIMPLEMENTED(0xD1FB50DC)


    override fun registerModule() {
        registerFunctionInt("sceKernelRegisterExitCallback", 0x4AC57943, since = 150) {
            sceKernelRegisterExitCallback(
                int
            )
        }
        registerFunctionVoid("sceKernelExitGame", 0x05572A5F, since = 150) { sceKernelExitGame(thread) }

        registerFunctionRaw("sceKernelExitGameWithStatus", 0x2AC9954B, since = 150) { sceKernelExitGameWithStatus(it) }
        registerFunctionRaw("LoadExecForUser_362A956B", 0x362A956B, since = 150) { LoadExecForUser_362A956B(it) }
        registerFunctionRaw("LoadExecForUser_8ADA38D3", 0x8ADA38D3, since = 150) { LoadExecForUser_8ADA38D3(it) }
        registerFunctionRaw("sceKernelLoadExec", 0xBD2F1094, since = 150) { sceKernelLoadExec(it) }
        registerFunctionRaw("LoadExecForUser_D1FB50DC", 0xD1FB50DC, since = 150) { LoadExecForUser_D1FB50DC(it) }
    }
}
