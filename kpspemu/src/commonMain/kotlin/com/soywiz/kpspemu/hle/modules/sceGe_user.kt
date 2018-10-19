package com.soywiz.kpspemu.hle.modules

import com.soywiz.kpspemu.*
import com.soywiz.kpspemu.cpu.*
import com.soywiz.kpspemu.ge.*
import com.soywiz.kpspemu.hle.*
import com.soywiz.kpspemu.hle.error.*
import com.soywiz.kpspemu.mem.*
import com.soywiz.kpspemu.util.*

class sceGe_user(emulator: Emulator) : SceModule(emulator, "sceGe_user", 0x40010011, "ge.prx", "sceGE_Manager") {
    fun sceGeEdramGetAddr(): Int = 0x04000000

    val geCallbacks = ResourceList("GeCallback") { GeCallback(it) }

    fun sceGeSetCallback(callbackDataPtr: Ptr): Int {
        return geCallbacks.alloc().apply {
            signal_func = callbackDataPtr.lw(0)
            signal_arg = callbackDataPtr.lw(4)
            finish_func = callbackDataPtr.lw(8)
            finish_arg = callbackDataPtr.lw(12)
        }.id
    }

    fun sceGeUnsetCallback(callbackId: Int): Int {
        geCallbacks.freeById(callbackId)
        return 0
    }

    fun sceGeListEnQueue(start: Ptr, stall: Ptr, callbackId: Int, pspGeListArgs: Ptr): Int {
        return ge.listEnqueue(start.addr, stall.addr, geCallbacks[callbackId], pspGeListArgs.addr).id
    }

    private fun getList(displayListId: Int): GeList {
        return ge.lists.tryGetById(displayListId) ?: sceKernelException(-1)
    }

    fun sceGeListUpdateStallAddr(displayListId: Int, stall: Ptr): Int {
        //println("WIP: sceGeListUpdateStallAddr")
        val list = getList(displayListId)
        list.stall = stall.addr
        list.run()
        return 0
    }

    fun sceGeListSync(displayListId: Int, syncType: Int): Int {
        //println("WIP: sceGeListSync")
        val displayList = getList(displayListId)
        //thread.suspend(WaitObject.PROMISE(displayList.syncAsync(syncType)), cb = false)
        displayList.sync(syncType)
        return 0
    }

    suspend fun sceGeDrawSync(syncType: Int): Int {
        //println("WIP: sceGeDrawSync")
        //thread.suspend(WaitObject.PROMISE(ge.syncAsync(syncType)), cb = false)
        ge.sync(syncType)
        emulator.gpu.flush()
        emulator.gpuRenderer.tryExecuteNow()
        when (syncType) {
            PspGeSyncType.PSP_GE_LIST_DONE, PspGeSyncType.PSP_GE_LIST_DRAWING_DONE -> {
                emulator.gpuRenderer.queuedJobs.waitValue(0)
            }
            else -> Unit // @TODO: @BUG in JS: See https://youtrack.jetbrains.com/issue/KT-22544
        }
        //display.waitVblank()
        return 0
    }

    fun sceGeSaveContext(ptr: Ptr): Int {
        for (n in 0 until GeState.STATE_NWORDS) ptr.sw(n * 4, ge.state.data[n])
        return 0
    }

    fun sceGeRestoreContext(ptr: Ptr): Int {
        for (n in 0 until GeState.STATE_NWORDS) ge.state.data[n] = ptr.lw(n * 4)
        return 0
    }

    fun sceGeEdramGetSize(): Int = 0x00200000; // 2MB

    fun sceGeListEnQueueHead(cpu: CpuState): Unit = UNIMPLEMENTED(0x1C0D95A6)
    fun sceGeContinue(cpu: CpuState): Unit = UNIMPLEMENTED(0x4C06E472)
    fun sceGeGetMtx(cpu: CpuState): Unit = UNIMPLEMENTED(0x57C8945B)
    fun sceGeListDeQueue(cpu: CpuState): Unit = UNIMPLEMENTED(0x5FB86AB0)
    fun sceGeBreak(cpu: CpuState): Unit = UNIMPLEMENTED(0xB448EC0D)
    fun sceGeEdramSetAddrTranslation(cpu: CpuState): Unit = UNIMPLEMENTED(0xB77905EA)
    fun sceGeGetCmd(cpu: CpuState): Unit = UNIMPLEMENTED(0xDC93CFEF)
    fun sceGeGetStack(cpu: CpuState): Unit = UNIMPLEMENTED(0xE66CB92E)

    override fun registerModule() {
        // Address
        registerFunctionInt("sceGeEdramGetAddr", 0xE47E40E4, since = 150) { sceGeEdramGetAddr() }
        registerFunctionInt("sceGeEdramGetSize", 0x1F6752AD, since = 150) { sceGeEdramGetSize() }

        // Lists & Sync
        registerFunctionInt("sceGeListUpdateStallAddr", 0xE0D68148, since = 150) { sceGeListUpdateStallAddr(int, ptr) }
        registerFunctionInt("sceGeListEnQueue", 0xAB49E76A, since = 150) { sceGeListEnQueue(ptr, ptr, int, ptr) }
        registerFunctionInt("sceGeListSync", 0x03444EB4, since = 150) { sceGeListSync(int, int) }
        registerFunctionSuspendInt("sceGeDrawSync", 0xB287BD61, since = 150) { sceGeDrawSync(int) }

        // Callbacks
        registerFunctionInt("sceGeSetCallback", 0xA4FC06A4, since = 150) { sceGeSetCallback(ptr) }
        registerFunctionInt("sceGeUnsetCallback", 0x05DB22CE, since = 150) { sceGeUnsetCallback(int) }

        // Context
        registerFunctionInt("sceGeSaveContext", 0x438A385A, since = 150) { sceGeSaveContext(ptr) }
        registerFunctionInt("sceGeRestoreContext", 0x0BF608FB, since = 150) { sceGeRestoreContext(ptr) }

        // Unimplemented
        registerFunctionRaw("sceGeListEnQueueHead", 0x1C0D95A6, since = 150) { sceGeListEnQueueHead(it) }
        registerFunctionRaw("sceGeContinue", 0x4C06E472, since = 150) { sceGeContinue(it) }
        registerFunctionRaw("sceGeGetMtx", 0x57C8945B, since = 150) { sceGeGetMtx(it) }
        registerFunctionRaw("sceGeListDeQueue", 0x5FB86AB0, since = 150) { sceGeListDeQueue(it) }
        registerFunctionRaw("sceGeBreak", 0xB448EC0D, since = 150) { sceGeBreak(it) }
        registerFunctionRaw(
            "sceGeEdramSetAddrTranslation",
            0xB77905EA,
            since = 150
        ) { sceGeEdramSetAddrTranslation(it) }
        registerFunctionRaw("sceGeGetCmd", 0xDC93CFEF, since = 150) { sceGeGetCmd(it) }
        registerFunctionRaw("sceGeGetStack", 0xE66CB92E, since = 150) { sceGeGetStack(it) }
    }
}
