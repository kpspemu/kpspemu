package com.soywiz.kpspemu.hle.modules

import com.soywiz.korio.async.*
import com.soywiz.kpspemu.*
import com.soywiz.kpspemu.cpu.*
import com.soywiz.kpspemu.hle.*
import com.soywiz.kpspemu.hle.error.*
import com.soywiz.kpspemu.hle.manager.*
import com.soywiz.kpspemu.mem.*
import com.soywiz.kpspemu.util.*
import com.soywiz.krypto.encoding.*

class ThreadManForUser_Thread(val tmodule: ThreadManForUser) : SceSubmodule<ThreadManForUser>(tmodule) {
    val logger get() = tmodule.logger

    private fun thread(id: Int): PspThread =
        threadManager.tryGetById(id) ?: throw SceKernelException(SceKernelErrors.ERROR_KERNEL_NOT_FOUND_THREAD)

    fun sceKernelCreateThread(
        name: String?,
        entryPoint: Int,
        initPriority: Int,
        stackSize: Int,
        attributes: Int,
        optionPtr: Ptr
    ): Int {
        //println("sceKernelCreateThread: '$name', ${entryPoint.hex}, $initPriority, $stackSize, ${attributes.hex}, $optionPtr")
        logger.trace { "sceKernelCreateThread: '$name', ${entryPoint.hex}, $initPriority, $stackSize, ${attributes.hex}, $optionPtr" }
        val thread = threadManager.create(name ?: "unknown", entryPoint, initPriority, stackSize, attributes, optionPtr)
        val k0Struct = K0Structure(
            threadId = thread.id,
            stackAddr = thread.stack.low.toInt(),
            f1 = -1,
            f2 = -1
        )
        mem.sw(thread.stack.low.toInt(), thread.id)

        thread.state.K0 = thread.putDataInStack(K0Structure.toByteArray(k0Struct)).low

        //println("sceKernelCreateThread: ${thread.id}")
        //println("thread.id = ${thread.id}")
        return thread.id
    }

    fun sceKernelStartThread(currentThread: PspThread, threadId: Int, userDataLength: Int, userDataPtr: Ptr): Unit {
        //println("sceKernelStartThread: $threadId, $userDataLength, $userDataPtr")
        logger.trace { "sceKernelStartThread: $threadId, $userDataLength, $userDataPtr" }
        //println("sceKernelStartThread: $threadId")
        val thread = thread(threadId)
        if (userDataPtr.isNotNull) {
            val localUserDataPtr = thread.putDataInStack(userDataPtr.readBytes(userDataLength))
            thread.state.r4 = userDataLength
            thread.state.r5 = localUserDataPtr.addr
        } else {
            thread.state.r4 = 0
            thread.state.r5 = 0
        }
        thread.state.GP = currentThread.state.GP
        thread.start()
        threadManager.suspendReturnInt(0)
    }

    fun sceKernelGetThreadCurrentPriority(thread: PspThread): Int = thread.priority

    suspend fun _sceKernelSleepThread(currentThread: PspThread, cb: Boolean): Int {
        currentThread.onWakeUp.waitOne()
        return 0
    }

    suspend fun sceKernelSleepThread(currentThread: PspThread): Int = _sceKernelSleepThread(currentThread, cb = false)
    suspend fun sceKernelSleepThreadCB(currentThread: PspThread): Int = _sceKernelSleepThread(currentThread, cb = true)

    suspend fun _sceKernelDelayThread(thread: PspThread, microseconds: Int, cb: Boolean): Int {
        thread.sleepMicro(microseconds)
        return 0
    }

    suspend fun sceKernelDelayThreadCB(thread: PspThread, microseconds: Int): Int =
        _sceKernelDelayThread(thread, microseconds, cb = true)

    suspend fun sceKernelDelayThread(thread: PspThread, microseconds: Int): Int =
        _sceKernelDelayThread(thread, microseconds, cb = false)


    suspend fun _sceKernelWaitThreadEnd(currentThread: PspThread, threadId: Int, timeout: Ptr, cb: Boolean): Int {
        val thread = thread(threadId)
        currentThread.waitInfo = threadId
        thread.onEnd.add {
            println("ENDED!")
        }
        thread.onEnd.waitOne()
        println("Resumed!")
        return 0
    }

    suspend fun sceKernelWaitThreadEnd(currentThread: PspThread, threadId: Int, timeout: Ptr): Int =
        _sceKernelWaitThreadEnd(currentThread, threadId, timeout, cb = false)

    suspend fun sceKernelWaitThreadEndCB(currentThread: PspThread, threadId: Int, timeout: Ptr): Int =
        _sceKernelWaitThreadEnd(currentThread, threadId, timeout, cb = true)

    fun sceKernelReferThreadStatus(currentThread: PspThread, threadId: Int, out: Ptr): Int {
        val actualThreadId = if (threadId == -1) currentThread.id else threadId
        val thread = thread(actualThreadId)

        val info = SceKernelThreadInfo()

        info.size = SceKernelThreadInfo.size

        info.name = thread.name
        info.attributes = thread.attributes
        info.status = thread.status
        info.threadPreemptionCount = thread.preemptionCount
        info.entryPoint = thread.entryPoint
        info.stackPointer = thread.stack.high.toInt()
        info.stackSize = thread.stack.size.toInt()
        info.GP = thread.state.GP

        info.priorityInit = thread.initPriority
        info.priority = thread.priority
        info.waitType = 0
        info.waitId = 0
        info.wakeupCount = 0
        info.exitStatus = thread.exitStatus
        info.runClocksLow = 0
        info.runClocksHigh = 0
        info.interruptPreemptionCount = 0
        info.threadPreemptionCount = 0
        info.releaseCount = 0

        out.write(SceKernelThreadInfo, info)

        return 0
    }

    fun sceKernelGetThreadId(thread: PspThread): Int = thread.id

    fun sceKernelTerminateThread(threadId: Int): Int {
        val newThread = thread(threadId)
        newThread.stop("_sceKernelTerminateThread")
        newThread.exitStatus = 0x800201ac.toInt()
        return 0
    }

    fun sceKernelDeleteThread(id: Int): Int {
        println("sceKernelDeleteThread($id)")
        val thread = thread(id)
        thread.delete()
        return 0
    }

    fun sceKernelExitThread(thread: PspThread, exitStatus: Int): Unit {
        println("sceKernelExitThread($thread) exitStatus=$exitStatus")
        thread.exitStatus = exitStatus
        thread.stop()
        threadManager.suspendReturnVoid()
    }

    fun sceKernelChangeCurrentThreadAttr(currentThread: PspThread, removeAttributes: Int, addAttributes: Int): Int {
        currentThread.attributes = (currentThread.attributes and removeAttributes.inv()) or addAttributes
        return 0
    }

    fun sceKernelChangeThreadPriority(id: Int, priority: Int): Int {
        val thread = thread(id)
        thread.threadManager.setThreadPriority(thread, priority)
        return 0
    }

    fun sceKernelTerminateDeleteThread(id: Int): Int {
        sceKernelTerminateThread(id)
        sceKernelDeleteThread(id)
        return 0
    }

    fun sceKernelGetThreadExitStatus(threadId: Int): Int = thread(threadId).exitStatus

    fun sceKernelRegisterThreadEventHandler(cpu: CpuState): Unit = UNIMPLEMENTED(0x0C106E53)
    fun sceKernelDelaySysClockThreadCB(cpu: CpuState): Unit = UNIMPLEMENTED(0x1181E963)
    fun sceKernelDonateWakeupThread(cpu: CpuState): Unit = UNIMPLEMENTED(0x1AF94D03)
    fun sceKernelResumeDispatchThread(cpu: CpuState): Unit = UNIMPLEMENTED(0x27E22EC2)
    fun sceKernelReleaseWaitThread(cpu: CpuState): Unit = UNIMPLEMENTED(0x2C34E053)
    fun sceKernelReferThreadEventHandlerStatus(cpu: CpuState): Unit = UNIMPLEMENTED(0x369EEB6B)
    fun sceKernelSuspendDispatchThread(cpu: CpuState): Unit = UNIMPLEMENTED(0x3AD58B8C)
    fun sceKernelGetThreadStackFreeSize(cpu: CpuState): Unit = UNIMPLEMENTED(0x52089CA1)
    fun _sceKernelExitThread(cpu: CpuState): Unit = UNIMPLEMENTED(0x532A522E)
    fun sceKernelGetThreadmanIdType(cpu: CpuState): Unit = UNIMPLEMENTED(0x57CF62DD)
    fun sceKernelReferThreadProfiler(cpu: CpuState): Unit = UNIMPLEMENTED(0x64D4540E)
    fun sceKernelResumeThread(cpu: CpuState): Unit = UNIMPLEMENTED(0x75156E8F)
    fun sceKernelExitDeleteThread(cpu: CpuState): Unit = UNIMPLEMENTED(0x809CE29B)
    fun sceKernelRotateThreadReadyQueue(cpu: CpuState): Unit = UNIMPLEMENTED(0x912354A7)
    fun sceKernelGetThreadmanIdList(cpu: CpuState): Unit = UNIMPLEMENTED(0x94416130)
    fun sceKernelDelaySysClockThread(cpu: CpuState): Unit = UNIMPLEMENTED(0xBD123D9E)
    fun sceKernelSuspendThread(cpu: CpuState): Unit = UNIMPLEMENTED(0x9944F31F)
    fun sceKernelSleepThread(cpu: CpuState): Unit = UNIMPLEMENTED(0x9ACE131E)
    fun sceKernelCheckThreadStack(cpu: CpuState): Unit = UNIMPLEMENTED(0xD13BDE95)
    fun sceKernelWakeupThread(cpu: CpuState): Unit = UNIMPLEMENTED(0xD59EAD2F)
    fun sceKernelCancelWakeupThread(cpu: CpuState): Unit = UNIMPLEMENTED(0xFCCFAD26)
    fun sceKernelReferThreadRunStatus(cpu: CpuState): Unit = UNIMPLEMENTED(0xFFC36A14)

    fun registerSubmodule() = tmodule.apply {
        registerFunctionInt("sceKernelCreateThread", 0x446D8DE6, since = 150) {
            sceKernelCreateThread(
                str,
                int,
                int,
                int,
                int,
                ptr
            )
        }
        registerFunctionVoid("sceKernelStartThread", 0xF475845D, since = 150) {
            sceKernelStartThread(
                thread,
                int,
                int,
                ptr
            )
        }
        registerFunctionInt(
            "sceKernelGetThreadCurrentPriority",
            0x94AA61EE,
            since = 150
        ) { sceKernelGetThreadCurrentPriority(thread) }
        registerFunctionSuspendInt("sceKernelSleepThread", 0x9ACE131E, since = 150) { sceKernelSleepThread(thread) }
        registerFunctionSuspendInt("sceKernelSleepThreadCB", 0x82826F70, since = 150) { sceKernelSleepThreadCB(thread) }
        registerFunctionSuspendInt("sceKernelDelayThreadCB", 0x68DA9E36, since = 150) {
            sceKernelDelayThreadCB(
                thread,
                int
            )
        }
        registerFunctionSuspendInt("sceKernelDelayThread", 0xCEADEB47, since = 150) {
            sceKernelDelayThread(
                thread,
                int
            )
        }
        registerFunctionSuspendInt("sceKernelWaitThreadEnd", 0x278C0DF5, since = 150) {
            sceKernelWaitThreadEnd(
                thread,
                int,
                ptr
            )
        }
        registerFunctionSuspendInt("sceKernelWaitThreadEndCB", 0x840E8133, since = 150) {
            sceKernelWaitThreadEndCB(
                thread,
                int,
                ptr
            )
        }
        registerFunctionInt("sceKernelReferThreadStatus", 0x17C1684E, since = 150) {
            sceKernelReferThreadStatus(
                thread,
                int,
                ptr
            )
        }
        registerFunctionInt("sceKernelGetThreadId", 0x293B45B8, since = 150) { sceKernelGetThreadId(thread) }
        registerFunctionInt("sceKernelTerminateThread", 0x616403BA, since = 150) { sceKernelTerminateThread(int) }
        registerFunctionInt("sceKernelDeleteThread", 0x9FA03CD3, since = 150) { sceKernelDeleteThread(int) }
        registerFunctionVoid("sceKernelExitThread", 0xAA73C935, since = 150) { sceKernelExitThread(thread, int) }
        registerFunctionInt(
            "sceKernelChangeCurrentThreadAttr",
            0xEA748E31,
            since = 150
        ) { sceKernelChangeCurrentThreadAttr(thread, int, int) }
        registerFunctionInt("sceKernelChangeThreadPriority", 0x71BC9871, since = 150) {
            sceKernelChangeThreadPriority(
                int,
                int
            )
        }
        registerFunctionInt("sceKernelTerminateDeleteThread", 0x383F7BCC, since = 150) {
            sceKernelTerminateDeleteThread(
                int
            )
        }
        registerFunctionInt(
            "sceKernelGetThreadExitStatus",
            0x3B183E26,
            since = 150
        ) { sceKernelGetThreadExitStatus(int) }

        registerFunctionRaw(
            "sceKernelRegisterThreadEventHandler",
            0x0C106E53,
            since = 150
        ) { sceKernelRegisterThreadEventHandler(it) }
        registerFunctionRaw("sceKernelDelaySysClockThreadCB", 0x1181E963, since = 150) {
            sceKernelDelaySysClockThreadCB(
                it
            )
        }
        registerFunctionRaw("sceKernelDonateWakeupThread", 0x1AF94D03, since = 150) { sceKernelDonateWakeupThread(it) }
        registerFunctionRaw(
            "sceKernelResumeDispatchThread",
            0x27E22EC2,
            since = 150
        ) { sceKernelResumeDispatchThread(it) }
        registerFunctionRaw("sceKernelReleaseWaitThread", 0x2C34E053, since = 150) { sceKernelReleaseWaitThread(it) }
        registerFunctionRaw(
            "sceKernelReferThreadEventHandlerStatus",
            0x369EEB6B,
            since = 150
        ) { sceKernelReferThreadEventHandlerStatus(it) }
        registerFunctionRaw("sceKernelSuspendDispatchThread", 0x3AD58B8C, since = 150) {
            sceKernelSuspendDispatchThread(
                it
            )
        }
        registerFunctionRaw(
            "sceKernelGetThreadStackFreeSize",
            0x52089CA1,
            since = 150
        ) { sceKernelGetThreadStackFreeSize(it) }
        registerFunctionRaw("_sceKernelExitThread", 0x532A522E, since = 150) { _sceKernelExitThread(it) }
        registerFunctionRaw("sceKernelGetThreadmanIdType", 0x57CF62DD, since = 150) { sceKernelGetThreadmanIdType(it) }
        registerFunctionRaw(
            "sceKernelReferThreadProfiler",
            0x64D4540E,
            since = 150
        ) { sceKernelReferThreadProfiler(it) }
        registerFunctionRaw("sceKernelResumeThread", 0x75156E8F, since = 150) { sceKernelResumeThread(it) }
        registerFunctionRaw("sceKernelExitDeleteThread", 0x809CE29B, since = 150) { sceKernelExitDeleteThread(it) }
        registerFunctionRaw(
            "sceKernelRotateThreadReadyQueue",
            0x912354A7,
            since = 150
        ) { sceKernelRotateThreadReadyQueue(it) }
        registerFunctionRaw("sceKernelGetThreadmanIdList", 0x94416130, since = 150) { sceKernelGetThreadmanIdList(it) }
        registerFunctionRaw("sceKernelSuspendThread", 0x9944F31F, since = 150) { sceKernelSuspendThread(it) }
        registerFunctionRaw(
            "sceKernelDelaySysClockThread",
            0xBD123D9E,
            since = 150
        ) { sceKernelDelaySysClockThread(it) }
        registerFunctionRaw("sceKernelCheckThreadStack", 0xD13BDE95, since = 150) { sceKernelCheckThreadStack(it) }
        registerFunctionRaw("sceKernelWakeupThread", 0xD59EAD2F, since = 150) { sceKernelWakeupThread(it) }
        registerFunctionRaw("sceKernelCancelWakeupThread", 0xFCCFAD26, since = 150) { sceKernelCancelWakeupThread(it) }
        registerFunctionRaw(
            "sceKernelReferThreadRunStatus",
            0xFFC36A14,
            since = 150
        ) { sceKernelReferThreadRunStatus(it) }
    }
}
