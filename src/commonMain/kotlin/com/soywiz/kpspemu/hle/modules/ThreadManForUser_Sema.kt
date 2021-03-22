package com.soywiz.kpspemu.hle.modules

import com.soywiz.korio.async.*
import com.soywiz.kpspemu.*
import com.soywiz.kpspemu.cpu.*
import com.soywiz.kpspemu.hle.*
import com.soywiz.kpspemu.hle.error.*
import com.soywiz.kpspemu.hle.manager.*
import com.soywiz.kpspemu.mem.*
import com.soywiz.kpspemu.util.*
import kotlin.math.*

class ThreadManForUser_Sema(val tmodule: ThreadManForUser) : SceSubmodule<ThreadManForUser>(tmodule) {
    class PspSemaphore(
        manager: SemaphoreManager, id: Int, name: String
    ) : Resource(manager, id, name) {
        var waitingThreads: Int = 0
        var attribute: Int = 0
        var initialCount: Int = 0
        var count: Int = 0
        var maxCount: Int = 0
        var active: Boolean = false
        val signal = Signal<Unit>()
    }

    class SemaphoreManager(emulator: Emulator) : Manager<PspSemaphore>("SemaphoreManager", emulator)

    val semaphoreManager by lazy { SemaphoreManager(emulator) }

    fun sceKernelCreateSema(name: String?, attribute: Int, initialCount: Int, maxCount: Int, options: Ptr): Int {
        val id = semaphoreManager.allocId()
        val sema = semaphoreManager.put(PspSemaphore(semaphoreManager, id, name ?: "sema$id"))
        sema.attribute = attribute
        sema.initialCount = initialCount
        sema.count = initialCount
        sema.maxCount = maxCount
        sema.signal.clear()
        sema.active = true
        return sema.id
    }

    suspend fun _sceKernelWaitSema(currentThread: PspThread, id: Int, expectedCount: Int, timeout: Ptr): Int {
        val sema = semaphoreManager.getById(id)
        while (sema.active && sema.count < expectedCount) {
            sema.waitingThreads++
            try {
                sema.signal.waitOne()
            } finally {
                sema.waitingThreads--
            }
        }
        sema.count -= expectedCount
        return 0
    }

    fun sceKernelSignalSema(currentThread: PspThread, id: Int, signal: Int): Int {
        val sema = semaphoreManager.getById(id)

        sema.count = min(sema.maxCount, sema.count + signal)
        sema.signal(Unit)

        return 0
    }

    fun sceKernelDeleteSema(id: Int): Int {
        val sema = semaphoreManager.getById(id)
        sema.active = false
        sema.signal(Unit)
        sema.free()
        return 0
    }

    fun sceKernelReferSemaStatus(id: Int, infoStream: Ptr): Int {
        val semaphore: PspSemaphore =
            semaphoreManager.tryGetById(id) ?: sceKernelException(SceKernelErrors.ERROR_KERNEL_NOT_FOUND_SEMAPHORE)
        val semaphoreInfo = SceKernelSemaInfo()
        semaphoreInfo.size = SceKernelSemaInfo.size
        semaphoreInfo.attributes = semaphore.attribute
        semaphoreInfo.currentCount = semaphore.count
        semaphoreInfo.initialCount = semaphore.initialCount
        semaphoreInfo.maximumCount = semaphore.maxCount
        semaphoreInfo.name = semaphore.name
        semaphoreInfo.numberOfWaitingThreads = semaphore.waitingThreads
        infoStream.write(SceKernelSemaInfo, semaphoreInfo)
        return 0
    }

    fun sceKernelPollSema(cpu: CpuState): Unit = UNIMPLEMENTED(0x58B1F937)
    fun sceKernelCancelSema(cpu: CpuState): Unit = UNIMPLEMENTED(0x8FFDF9A2)

    fun registerSubmodule() = tmodule.apply {
        registerFunctionInt("sceKernelCreateSema", 0xD6DA4BA1, since = 150) {
            sceKernelCreateSema(
                str,
                int,
                int,
                int,
                ptr
            )
        }
        registerFunctionSuspendInt(
            "sceKernelWaitSema",
            0x4E3A1105,
            since = 150,
            cb = false
        ) { _sceKernelWaitSema(thread, int, int, ptr) }
        registerFunctionSuspendInt("sceKernelWaitSemaCB", 0x6D212BAC, since = 150, cb = true) {
            _sceKernelWaitSema(
                thread,
                int,
                int,
                ptr
            )
        }
        registerFunctionInt("sceKernelSignalSema", 0x3F53E640, since = 150) { sceKernelSignalSema(thread, int, int) }
        registerFunctionInt("sceKernelDeleteSema", 0x28B6489C, since = 150) { sceKernelDeleteSema(int) }
        registerFunctionInt("sceKernelReferSemaStatus", 0xBC6FEBC5, since = 150) { sceKernelReferSemaStatus(int, ptr) }
        registerFunctionRaw("sceKernelPollSema", 0x58B1F937, since = 150) { sceKernelPollSema(it) }
        registerFunctionRaw("sceKernelCancelSema", 0x8FFDF9A2, since = 150) { sceKernelCancelSema(it) }
    }
}
