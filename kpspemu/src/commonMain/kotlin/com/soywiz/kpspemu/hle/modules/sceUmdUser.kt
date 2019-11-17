package com.soywiz.kpspemu.hle.modules

import com.soywiz.korio.async.*
import com.soywiz.kpspemu.*
import com.soywiz.kpspemu.cpu.*
import com.soywiz.kpspemu.hle.*
import com.soywiz.kpspemu.hle.error.*
import com.soywiz.kpspemu.util.*
import kotlinx.coroutines.*

@Suppress("ClassName", "UNUSED_PARAMETER")
class sceUmdUser(emulator: Emulator) : SceModule(emulator, "sceUmdUser", 0x40010011, "np9660.prx", "sceNp9660_driver") {
    val PSP_UMD_INIT = 0
    val PSP_UMD_NOT_PRESENT = 1 shl 0
    val PSP_UMD_PRESENT = 1 shl 1
    val PSP_UMD_CHANGED = 1 shl 2
    val PSP_UMD_NOT_READY = 1 shl 3
    val PSP_UMD_READY = 1 shl 4
    val PSP_UMD_READABLE = 1 shl 5

    val statusEvent = EventStatus {
        var out = PSP_UMD_PRESENT or PSP_UMD_READY
        if (activated) out = out or PSP_UMD_READABLE
        out
    }
    val status by statusEvent

    private var inserted: Boolean = true
        set(value) {
            field = value
            statusEvent.updated()
        }

    private var activated: Boolean = false
        set(value) {
            field = value
            statusEvent.updated()
        }

    fun sceUmdCheckMedium(): Int {
        return if (inserted) 1 else 0
    }

    fun sceUmdActivate(mode: Int, device: String): Int {
        activated = true
        return 0
    }

    fun sceUmdDeactivate(mode: Int, device: String): Int {
        activated = false
        return 0
    }

    suspend fun _sceUmdWaitDriveStatTimeout(stat: Int, timeout: Int): Int {
        try {
            //println("_sceUmdWaitDriveStatTimeout: ${stat.toStringUnsigned(2)} - ${status.toStringUnsigned(2)} - $timeout")
            emulator.coroutineContext.withOptTimeout(if (timeout < 0) null else timeout.toLong() / 1000, "_sceUmdWaitDriveStat") {
                statusEvent.waitAllBits(stat)
            }
            return 0
        } catch (e: CancellationException) {
            return SceKernelErrors.ERROR_KERNEL_WAIT_TIMEOUT
        }
    }

    suspend fun sceUmdWaitDriveStat(stat: Int): Int = _sceUmdWaitDriveStatTimeout(stat, -1)
    suspend fun sceUmdWaitDriveStatCB(stat: Int, timeout: Int): Int = _sceUmdWaitDriveStatTimeout(stat, timeout)
    suspend fun sceUmdWaitDriveStatWithTimer(stat: Int, timeout: Int): Int = _sceUmdWaitDriveStatTimeout(stat, timeout)
    fun sceUmdGetDriveStat(): Int = status

    fun sceUmdRegisterUMDCallBack(callbackId: Int): Int {
        logger.error { "sceUmdRegisterUMDCallBack UNIMPLEMENTED" }
        return 0
    }

    fun sceUmdUnRegisterUMDCallBack(callbackId: Int): Int {
        logger.error { "sceUmdUnRegisterUMDCallBack UNIMPLEMENTED" }
        return 0
    }

    fun sceUmdGetErrorStat(cpu: CpuState): Unit = UNIMPLEMENTED(0x20628E6F)
    fun sceUmdGetDiscInfo(cpu: CpuState): Unit = UNIMPLEMENTED(0x340B7686)
    fun sceUmdCancelWaitDriveStat(cpu: CpuState): Unit = UNIMPLEMENTED(0x6AF9B50A)
    fun sceUmdReplaceProhibit(cpu: CpuState): Unit = UNIMPLEMENTED(0x87533940)
    fun sceUmdReplacePermit(cpu: CpuState): Unit = UNIMPLEMENTED(0xCBE9F02A)

    override fun registerModule() {
        registerFunctionInt("sceUmdCheckMedium", 0x46EBB729, since = 150) { sceUmdCheckMedium() }
        registerFunctionInt("sceUmdActivate", 0xC6183D47, since = 150) { sceUmdActivate(int, istr) }
        registerFunctionInt("sceUmdDeactivate", 0xE83742BA, since = 150) { sceUmdDeactivate(int, istr) }
        registerFunctionSuspendInt(
            "sceUmdWaitDriveStat",
            0x8EF08FCE,
            since = 150,
            cb = false
        ) { sceUmdWaitDriveStat(int) }
        registerFunctionSuspendInt("sceUmdWaitDriveStatCB", 0x4A9E5E29, since = 150, cb = true) {
            sceUmdWaitDriveStatCB(
                int,
                int
            )
        }
        registerFunctionSuspendInt(
            "sceUmdWaitDriveStatWithTimer",
            0x56202973,
            since = 150
        ) { sceUmdWaitDriveStatWithTimer(int, int) }
        registerFunctionInt("sceUmdGetDriveStat", 0x6B4A146C, since = 150) { sceUmdGetDriveStat() }
        registerFunctionInt("sceUmdRegisterUMDCallBack", 0xAEE7404D, since = 150) { sceUmdRegisterUMDCallBack(int) }
        registerFunctionInt("sceUmdUnRegisterUMDCallBack", 0xBD2BDE07, since = 150) { sceUmdUnRegisterUMDCallBack(int) }

        registerFunctionRaw("sceUmdGetErrorStat", 0x20628E6F, since = 150) { sceUmdGetErrorStat(it) }
        registerFunctionRaw("sceUmdGetDiscInfo", 0x340B7686, since = 150) { sceUmdGetDiscInfo(it) }
        registerFunctionRaw("sceUmdCancelWaitDriveStat", 0x6AF9B50A, since = 150) { sceUmdCancelWaitDriveStat(it) }
        registerFunctionRaw("sceUmdReplaceProhibit", 0x87533940, since = 150) { sceUmdReplaceProhibit(it) }
        registerFunctionRaw("sceUmdReplacePermit", 0xCBE9F02A, since = 150) { sceUmdReplacePermit(it) }
    }
}
