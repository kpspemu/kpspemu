package com.soywiz.kpspemu.hle.modules

import com.soywiz.kpspemu.*
import com.soywiz.kpspemu.cpu.*
import com.soywiz.kpspemu.hle.*
import com.soywiz.kpspemu.mem.*
import com.soywiz.krypto.encoding.*

@Suppress("MemberVisibilityCanPrivate", "UNUSED_PARAMETER")
class ModuleMgrForUser(emulator: Emulator) :
    SceModule(emulator, "ModuleMgrForUser", 0x40010011, "modulemgr.prx", "sceModuleManager") {
    fun sceKernelLoadModule(path: String, flags: Int, sceKernelLMOption: Ptr): Int {
        logger.error { "Not implemented sceKernelLoadModule: $path" }
        return 0x08900000 // Module address
    }

    fun sceKernelStartModule(
        moduleId: Int,
        argumentSize: Int,
        argumentPointer: Ptr,
        status: Ptr,
        sceKernelSMOption: Ptr
    ): Int {
        logger.error { "Not implemented sceKernelStartModule: $moduleId" }
        return 0
    }

    fun sceKernelGetModuleIdByAddress(addr: Int): Int {
        logger.error { "Not implemented sceKernelGetModuleIdByAddress: ${addr.hex}" }
        return 2
    }

    fun sceKernelGetModuleId(): Int {
        logger.error { "Not implemented sceKernelGetModuleId" }
        return 1
    }

    fun sceKernelLoadModuleBufferMs(cpu: CpuState): Unit = UNIMPLEMENTED(0x1196472E)
    fun sceKernelLoadModuleBufferApp(cpu: CpuState): Unit = UNIMPLEMENTED(0x24EC0641)
    fun sceKernelUnloadModule(cpu: CpuState): Unit = UNIMPLEMENTED(0x2E0911AA)
    fun sceKernelGetModuleIdList(cpu: CpuState): Unit = UNIMPLEMENTED(0x644395E2)
    fun sceKernelLoadModuleMs(cpu: CpuState): Unit = UNIMPLEMENTED(0x710F61B5)
    fun sceKernelQueryModuleInfo(cpu: CpuState): Unit = UNIMPLEMENTED(0x748CBED9)
    fun ModuleMgrForUser_8F2DF740(cpu: CpuState): Unit = UNIMPLEMENTED(0x8F2DF740)
    fun sceKernelLoadModuleByID(cpu: CpuState): Unit = UNIMPLEMENTED(0xB7F46618)
    fun sceKernelStopUnloadSelfModule(cpu: CpuState): Unit = UNIMPLEMENTED(0xCC1D3699)
    fun sceKernelStopModule(cpu: CpuState): Unit = UNIMPLEMENTED(0xD1FF982A)
    fun sceKernelGetModuleGPByAddress(cpu: CpuState): Unit = UNIMPLEMENTED(0xD2FBC957)
    fun sceKernelSelfStopUnloadModule(cpu: CpuState): Unit = UNIMPLEMENTED(0xD675EBB8)
    fun ModuleMgrForUser_E4C4211C(cpu: CpuState): Unit = UNIMPLEMENTED(0xE4C4211C)
    fun ModuleMgrForUser_F2D8D1B4(cpu: CpuState): Unit = UNIMPLEMENTED(0xF2D8D1B4)
    fun sceKernelLoadModuleBufferUsbWlan(cpu: CpuState): Unit = UNIMPLEMENTED(0xF9275D98)
    fun ModuleMgrForUser_FBE27467(cpu: CpuState): Unit = UNIMPLEMENTED(0xFBE27467)
    fun ModuleMgrForUser_FEF27DC1(cpu: CpuState): Unit = UNIMPLEMENTED(0xFEF27DC1)


    override fun registerModule() {
        registerFunctionInt("sceKernelLoadModule", 0x977DE386, since = 150) { sceKernelLoadModule(istr, int, ptr) }
        registerFunctionInt("sceKernelStartModule", 0x50F0C1EC, since = 150) {
            sceKernelStartModule(
                int,
                int,
                ptr,
                ptr,
                ptr
            )
        }
        registerFunctionInt("sceKernelGetModuleIdByAddress", 0xD8B73127, since = 150) {
            sceKernelGetModuleIdByAddress(
                int
            )
        }
        registerFunctionInt("sceKernelGetModuleId", 0xF0A26395, since = 150) { sceKernelGetModuleId() }

        registerFunctionRaw("sceKernelLoadModuleBufferMs", 0x1196472E, since = 150) { sceKernelLoadModuleBufferMs(it) }
        registerFunctionRaw(
            "sceKernelLoadModuleBufferApp",
            0x24EC0641,
            since = 150
        ) { sceKernelLoadModuleBufferApp(it) }
        registerFunctionRaw("sceKernelUnloadModule", 0x2E0911AA, since = 150) { sceKernelUnloadModule(it) }
        registerFunctionRaw("sceKernelGetModuleIdList", 0x644395E2, since = 150) { sceKernelGetModuleIdList(it) }
        registerFunctionRaw("sceKernelLoadModuleMs", 0x710F61B5, since = 150) { sceKernelLoadModuleMs(it) }
        registerFunctionRaw("sceKernelQueryModuleInfo", 0x748CBED9, since = 150) { sceKernelQueryModuleInfo(it) }
        registerFunctionRaw("ModuleMgrForUser_8F2DF740", 0x8F2DF740, since = 150) { ModuleMgrForUser_8F2DF740(it) }
        registerFunctionRaw("sceKernelLoadModuleByID", 0xB7F46618, since = 150) { sceKernelLoadModuleByID(it) }
        registerFunctionRaw(
            "sceKernelStopUnloadSelfModule",
            0xCC1D3699,
            since = 150
        ) { sceKernelStopUnloadSelfModule(it) }
        registerFunctionRaw("sceKernelStopModule", 0xD1FF982A, since = 150) { sceKernelStopModule(it) }
        registerFunctionRaw(
            "sceKernelGetModuleGPByAddress",
            0xD2FBC957,
            since = 150
        ) { sceKernelGetModuleGPByAddress(it) }
        registerFunctionRaw(
            "sceKernelSelfStopUnloadModule",
            0xD675EBB8,
            since = 150
        ) { sceKernelSelfStopUnloadModule(it) }
        registerFunctionRaw("ModuleMgrForUser_E4C4211C", 0xE4C4211C, since = 150) { ModuleMgrForUser_E4C4211C(it) }
        registerFunctionRaw("ModuleMgrForUser_F2D8D1B4", 0xF2D8D1B4, since = 150) { ModuleMgrForUser_F2D8D1B4(it) }
        registerFunctionRaw(
            "sceKernelLoadModuleBufferUsbWlan",
            0xF9275D98,
            since = 150
        ) { sceKernelLoadModuleBufferUsbWlan(it) }
        registerFunctionRaw("ModuleMgrForUser_FBE27467", 0xFBE27467, since = 150) { ModuleMgrForUser_FBE27467(it) }
        registerFunctionRaw("ModuleMgrForUser_FEF27DC1", 0xFEF27DC1, since = 150) { ModuleMgrForUser_FEF27DC1(it) }
    }
}
