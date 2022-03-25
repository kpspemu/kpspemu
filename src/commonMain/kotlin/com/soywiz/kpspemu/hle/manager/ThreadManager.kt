package com.soywiz.kpspemu.hle.manager

import com.soywiz.kds.*
import com.soywiz.klock.*
import com.soywiz.klogger.*
import com.soywiz.kmem.*
import com.soywiz.korio.async.*
import com.soywiz.korio.error.*
import com.soywiz.korio.lang.*
import com.soywiz.kpspemu.*
import com.soywiz.kpspemu.cpu.*
import com.soywiz.kpspemu.cpu.dynarec.*
import com.soywiz.kpspemu.cpu.interpreter.*
import com.soywiz.kpspemu.mem.*
import com.soywiz.kpspemu.util.*
import kotlin.collections.set
import kotlin.math.*
import com.soywiz.kmem.umod
import com.soywiz.krypto.encoding.*
import kotlinx.coroutines.channels.*
import com.soywiz.korio.error.invalidOp as invalidOp1

//const val INSTRUCTIONS_PER_STEP = 500_000
//const val INSTRUCTIONS_PER_STEP = 1_000_000
//const val INSTRUCTIONS_PER_STEP = 2_000_000
//const val INSTRUCTIONS_PER_STEP = 4_000_000
const val INSTRUCTIONS_PER_STEP = 5_000_000
//const val INSTRUCTIONS_PER_STEP = 10_000_000
//const val INSTRUCTIONS_PER_STEP = 100_000_000


class ThreadsWithPriority(val priority: Int) : Comparable<ThreadsWithPriority> {
    val threads = arrayListOf<PspThread>()

    var currentIndex = 0
    val runningThreads get() = threads.count { it.running }

    fun next(): PspThread? = when {
        threads.isNotEmpty() -> threads[currentIndex++ % threads.size]
        else -> null
    }

    fun nextRunning(): PspThread? {
        for (n in 0 until threads.size) {
            val n = next()
            if (n != null && n.running) return n
        }
        return null
    }

    override fun compareTo(other: ThreadsWithPriority): Int {
        return this.priority.compareTo(other.priority)
    }

    override fun toString(): String = "ThreadsWithPriority(priority=$priority)"
}

class ThreadManager(emulator: Emulator) : Manager<PspThread>("Thread", emulator) {
    val logger = Logger("ThreadManager").apply {
        //level = LogLevel.TRACE
    }
    val prioritiesByValue = LinkedHashMap<Int, ThreadsWithPriority>()
    val priorities = PriorityQueue<ThreadsWithPriority>()
    val currentPriorities = PriorityQueue<ThreadsWithPriority>()

    val threads get() = resourcesById.values
    val waitingThreads: Int get() = resourcesById.count { it.value.waiting }
    val activeThreads: Int get() = resourcesById.count { it.value.running }
    val totalThreads: Int get() = resourcesById.size
    val aliveThreadCount: Int get() = resourcesById.values.count { it.running || it.waiting }
    val onThreadChanged = Signal<PspThread>()
    val onThreadChangedChannel = Channel<PspThread>().broadcast()
    val waitThreadChanged = onThreadChangedChannel.openSubscription()
    var currentThread: PspThread? = null

    override fun reset() {
        super.reset()
        currentThread = null
    }

    fun setThreadPriority(thread: PspThread, priority: Int) {
        if (thread.priority == priority) return

        val oldThreadsWithPriority = thread.threadsWithPriority
        if (oldThreadsWithPriority != null) {
            oldThreadsWithPriority.threads.remove(thread)
            if (oldThreadsWithPriority.threads.isEmpty()) {
                prioritiesByValue.remove(oldThreadsWithPriority.priority)
                priorities.remove(oldThreadsWithPriority)
            }
        }

        val newThreadsWithPriority = prioritiesByValue.getOrPut(priority) {
            ThreadsWithPriority(priority).also { priorities.add(it) }
        }
        thread.threadsWithPriority = newThreadsWithPriority
        newThreadsWithPriority.threads.add(thread)
    }

    fun create(
        name: String,
        entryPoint: Int,
        initPriority: Int,
        stackSize: Int,
        attributes: Int,
        optionPtr: Ptr
    ): PspThread {
        //priorities.sortBy { it.priority }
        var attr = attributes
        val ssize = max(stackSize, 0x200).nextAlignedTo(0x100)
        //val ssize = max(stackSize, 0x20000).nextAlignedTo(0x10000)
        val stack = memoryManager.stackPartition.allocateHigh(ssize, "${name}_stack")

        println(stack.toString2())

        val thread = PspThread(this, allocId(), name, entryPoint, stack, initPriority, attributes, optionPtr)
        logger.info { "stack:%08X-%08X (%d)".format(stack.low.toInt(), stack.high.toInt(), stack.size.toInt()) }

        //memoryManager.userPartition.dump()

        attr = attr or PspThreadAttributes.User
        attr = attr or PspThreadAttributes.LowFF

        if (!(attr hasFlag PspThreadAttributes.NoFillStack)) {
            logger.trace { "FILLING: $stack" }
            mem.fill(-1, stack.low.toInt(), stack.size.toInt())
        } else {
            logger.trace { "NOT FILLING: $stack" }
        }
        threadManager.onThreadChanged(thread)
        threadManager.onThreadChangedChannel.trySend(thread)
        return thread
    }

    @Deprecated("USE suspendReturnInt or suspendReturnVoid")
    fun suspend(): Nothing {
        throw CpuBreakExceptionCached(CpuBreakException.THREAD_WAIT)
    }

    fun suspendReturnInt(value: Int): Nothing {
        currentThread?.state?.V0 = value
        throw CpuBreakExceptionCached(CpuBreakException.THREAD_WAIT)
    }

    fun suspendReturnVoid(): Nothing {
        throw CpuBreakExceptionCached(CpuBreakException.THREAD_WAIT)
    }

    fun getActiveThreadPriorities(): List<Int> = threads.filter { it.running }.map { it.priority }.distinct().sorted()
    fun getActiveThreadsWithPriority(priority: Int): List<PspThread> =
        threads.filter { it.running && it.priority == priority }

    val lowestThreadPriority get() = threads.minByOrNull { it.priority }?.priority ?: 0

    fun getFirstThread(): PspThread? {
        for (thread in threads) {
            if (thread.priority == lowestThreadPriority) return thread
        }
        return null
    }

    fun getNextPriority(priority: Int): Int? {
        return getActiveThreadPriorities().firstOrNull { it > priority }
    }

    fun computeNextThread(prevThread: PspThread?): PspThread? {
        if (prevThread == null) return getFirstThread()
        val threadsWithPriority = getActiveThreadsWithPriority(prevThread.priority)
        val threadsWithPriorityCount = threadsWithPriority.size
        if (threadsWithPriorityCount > 0) {
            val index = threadsWithPriority.indexOf(prevThread) umod threadsWithPriorityCount
            return threadsWithPriority.getOrNull((index + 1) umod threadsWithPriorityCount)
        } else {
            val nextPriority = getNextPriority(prevThread.priority)
            return nextPriority?.let { getActiveThreadsWithPriority(it).firstOrNull() }
        }
    }

    suspend fun waitThreadChange() {
        //println("[1]")
        /*
        kotlinx.coroutines.withTimeout(16L) {
            waitThreadChanged.receive()
        }
         */
        delay(1.milliseconds)
        //onThreadChanged.waitOne()
        //println("[2]")
        //coroutineContext.sleep(0)
    }

    enum class StepResult {
        NO_THREAD, BREAKPOINT, TIMEOUT
    }

    fun step(): StepResult {
        val start = DateTime.now()

        if (emulator.globalTrace) {
            println("-----")
            for (thread in threads) {
                println("- ${thread.name} : ${thread.priority} : running=${thread.running}")
            }
            println("-----")
        }

        currentPriorities.clear()
        currentPriorities.addAll(priorities)

        //Console.error("ThreadManager.STEP")

        while (currentPriorities.isNotEmpty()) {
            val threads = currentPriorities.removeHead()
            //println("threads: $threads")
            while (true) {
                val now = DateTime.now()
                val current = threads.nextRunning() ?: break
                //println("    Running Thread.STEP: $current")
                if (emulator.globalTrace) println("Current thread: ${current.name}")
                try {
                    current.step(now.unixMillisDouble)
                } catch (e: BreakpointException) {
                    //Console.error("StepResult.BREAKPOINT: $e")
                    return StepResult.BREAKPOINT
                }
                if (emulator.globalTrace) println("Next thread: ${current.name}")

                val elapsed = now - start
                if (elapsed >= 16.milliseconds) {
                    //coroutineContext.delay(1.milliseconds)
                    Console.error("StepResult.TIMEOUT: $elapsed")
                    return StepResult.TIMEOUT
                }
            }
        }

        //dump()
        return StepResult.NO_THREAD
    }

    val availablePriorities: List<Int> get() = resourcesById.values.filter { it.running }.map { it.priority }.distinct().sorted()

    val availableThreadCount get() = resourcesById.values.count { it.running }
    val availableThreads get() = resourcesById.values.filter { it.running }.sortedBy { it.priority }

    val traces = hashMapOf<String, Boolean>()

    fun trace(name: String, trace: Boolean = true) {
        if (trace) {
            traces[name] = true
        } else {
            traces.remove(name)
        }
        tryGetByName(name)?.updateTrace()
    }

    fun stopAllThreads() {
        for (t in resourcesById.values.toList()) {
            t.exitAndKill()
        }
        throw CpuBreakExceptionCached(CpuBreakException.THREAD_EXIT_KILL)
    }

    fun executeInterrupt(address: Int, argument: Int) {
        val gcpustate = emulator.globalCpuState
        val oldInsideInterrupt = gcpustate.insideInterrupt
        gcpustate.insideInterrupt = true
        try {
            val thread = threads.first()
            val cpu = thread.state.clone()
            cpu._thread = thread
            val interpreter = CpuInterpreter(cpu, emulator.breakpoints, emulator.nameProvider)

            cpu.setPC(address)
            cpu.RA = CpuBreakException.INTERRUPT_RETURN_RA
            cpu.r4 = argument

            val start = timeManager.getTimeInMicroseconds()
            while (true) {
                val now = timeManager.getTimeInMicroseconds()
                interpreter.steps(10000)
                if (now - start >= 16.0) {
                    println("Interrupt is taking too long...")
                    break // Rest a bit
                }
            }
        } catch (e: CpuBreakException) {
            if (e.id != CpuBreakException.INTERRUPT_RETURN) {
                throw e
            }
        } finally {
            gcpustate.insideInterrupt = oldInsideInterrupt
        }
    }

    fun delayThread(micros: Int) {
        // @TODO:
    }

    fun dump() {
        println("ThreadManager.dump:")
        for (thread in threads) {
            println(" - $thread")
        }
    }

    val summary: String
        get() = "[" + threads.map { "'${it.name}'#${it.id} P${it.priority} : ${it.phase}" }.joinToString(", ") + "]"
}

sealed class WaitObject {
    //data class TIME(val instant: Double) : WaitObject() {
    //	override fun toString(): String = "TIME(${instant.toLong()})"
    //}
    //data class PROMISE(val promise: Promise<Unit>, val reason: String) : WaitObject()
    data class COROUTINE(val reason: String) : WaitObject()

    //object SLEEP : WaitObject()
    object VBLANK : WaitObject()
}

class PspThread internal constructor(
    val threadManager: ThreadManager,
    id: Int,
    name: String,
    val entryPoint: Int,
    val stack: MemoryPartition,
    val initPriority: Int,
    var attributes: Int,
    val optionPtr: Ptr
) : Resource(threadManager, id, name), WithEmulator {
    var threadsWithPriority: ThreadsWithPriority? = null

    var preemptionCount: Int = 0
    val totalExecutedInstructions: Long get() = state.totalExecuted
    val onEnd = Signal<Unit>()
    val onWakeUp = Signal<Unit>()
    val logger = Logger("PspThread")

    enum class Phase {
        STOPPED,
        RUNNING,
        WAITING,
        DELETED
    }

    override fun toString(): String {
        return "PspThread(id=$id, name='$name', phase=$phase, status=$status, priority=$priority, waitObject=$waitObject, waitInfo=$waitInfo)"
    }

    val status: Int
        get() {
            var out: Int = 0
            if (running) out = out or ThreadStatus.RUNNING
            if (waiting) out = out or ThreadStatus.WAIT
            if (phase == Phase.DELETED) out = out or ThreadStatus.DEAD
            return out
        }

    var acceptingCallbacks: Boolean = false
    var waitObject: WaitObject? = null
    var waitInfo: Any? = null
    var exitStatus: Int = 0
    var phase: Phase = Phase.STOPPED
        set(value) = run { field = value }

    val priority: Int get() = threadsWithPriority?.priority ?: Int.MAX_VALUE

    val running: Boolean get() = phase == Phase.RUNNING
    val waiting: Boolean get() = waitObject != null

    override val emulator get() = manager.emulator
    val state = CpuState("state.thread.$name", emulator.globalCpuState, emulator.syscalls).apply {
        _thread = this@PspThread
        setPC(entryPoint)
        SP = stack.high.toInt()
        RA = CpuBreakException.THREAD_EXIT_KIL_RA

        println("CREATED THREAD('$name'): PC=${PC.hex}, SP=${SP.hex}")
    }
    val interpreter = CpuInterpreter(state, emulator.breakpoints, emulator.nameProvider)
    val dynarek = DynarekRunner(state, emulator.breakpoints, emulator.nameProvider)
    //val interpreter = FastCpuInterpreter(state)

    init {
        updateTrace()
        setThreadProps(Phase.STOPPED)
        threadManager.setThreadPriority(this, initPriority)
    }

    fun setThreadProps(phase: Phase, waitObject: WaitObject? = this.waitObject, waitInfo: Any? = this.waitInfo, acceptingCallbacks: Boolean = this.acceptingCallbacks) {
        this.phase = phase
        this.waitObject = waitObject
        this.waitInfo = waitInfo
        this.acceptingCallbacks = acceptingCallbacks
        if (phase == Phase.STOPPED) {
            onEnd(Unit)
        }
        if (phase == Phase.DELETED) {
            manager.freeById(id)
            logger.warn { "Deleting Thread: $name" }
        }
        threadManager.onThreadChanged(this)
        threadManager.onThreadChangedChannel.trySend(this)
    }

    fun updateTrace() {
        interpreter.trace = threadManager.traces[name] == true
    }

    fun putDataInStack(bytes: ByteArray, alignment: Int = 0x10): PtrArray {
        val blockSize = bytes.size.nextAlignedTo(alignment)
        state.SP -= blockSize
        mem.write(state.SP, bytes)
        return PtrArray(mem.ptr(state.SP), bytes.size)
    }

    fun putWordInStack(word: Int, alignment: Int = 0x10): PtrArray {
        val blockSize = 4.nextAlignedTo(alignment)
        state.SP -= blockSize
        mem.sw(state.SP, word)
        return mem.ptr(state.SP).array(4)
    }

    fun putWordsInStack(vararg words: Int, alignment: Int = 0x10): PtrArray {
        val blockSize = (words.size * 4).nextAlignedTo(alignment)
        state.SP -= blockSize
        for (n in 0 until words.size) mem.sw(state.SP + n * 4, words[n])
        return mem.ptr(state.SP).array(words.size * 4)
    }

    fun start() {
        resume()
    }

    fun resume() {
        setThreadProps(phase = Phase.RUNNING, waitObject = null, waitInfo = null, acceptingCallbacks = false)
    }

    fun stop(reason: String = "generic") {
        if (phase != Phase.STOPPED) {
            logger.warn { "Stopping Thread: $name : reason=$reason" }
            setThreadProps(phase = Phase.STOPPED)
        }
        //threadManager.onThreadChanged(this)
    }

    fun delete() {
        stop()
        setThreadProps(phase = Phase.DELETED)
    }

    fun exitAndKill() {
        stop()
        delete()
    }

    fun step(now: Double, trace: Boolean = tracing): Int {
        //if (name == "update_thread") {
        //	println("Ignoring: Thread.${this.name}")
        //	stop("ignoring")
        //	return
        //}
        //println("Step: Thread.${this.name}")
        preemptionCount++
        try {
            if (emulator.interpreted) {
                interpreter.steps(INSTRUCTIONS_PER_STEP, trace)
            } else {
                dynarek.steps(INSTRUCTIONS_PER_STEP, trace)
            }
            return 0
        } catch (e: CpuBreakException) {
            when (e.id) {
                CpuBreakException.THREAD_EXIT_KILL -> {
                    logger.info { "BREAK: THREAD_EXIT_KILL ('${this.name}', ${this.id})" }
                    println("BREAK: THREAD_EXIT_KILL ('${this.name}', ${this.id})")
                    exitAndKill()
                }
                CpuBreakException.THREAD_WAIT -> {
                    // Do nothing
                }
                CpuBreakException.INTERRUPT_RETURN -> {
                    // Do nothing
                }
                else -> {
                    println("CPU: ${state.summary}")
                    println("ERROR at PspThread.step")
                    e.printStackTrace()
                    throw e
                }
            }
            return e.id
        }
    }

    fun markWaiting(wait: WaitObject, cb: Boolean) {
        setThreadProps(phase = Phase.WAITING, waitObject = wait, acceptingCallbacks = cb)
    }

    fun suspend(wait: WaitObject, cb: Boolean) {
        markWaiting(wait, cb)
        //if (wait is WaitObject.PROMISE) wait.promise.then { resume() }
        threadManager.suspend()
    }

    var pendingAccumulatedMicrosecondsToWait: Int = 0

    suspend fun sleepMicro(microseconds: Int) {
        val totalMicroseconds = pendingAccumulatedMicrosecondsToWait + microseconds
        pendingAccumulatedMicrosecondsToWait = totalMicroseconds % 1000
        val totalMilliseconds = totalMicroseconds / 1000

        // @TODO: This makes sceRtc test to be flaky
        //if (totalMilliseconds < 1) {
        //	pendingAccumulatedMicrosecondsToWait += totalMilliseconds * 1000
        //} else {
        coroutineContext.delay(totalMilliseconds.milliseconds)
        //}
    }

    suspend fun sleepSeconds(seconds: Double) = sleepMicro((seconds * 1_000_000).toInt())

    suspend fun sleepSecondsIfRequired(seconds: Double) {
        if (seconds > 0.0) sleepMicro((seconds * 1_000_000).toInt())
    }

    fun cpuBreakException(e: CpuBreakException) {
    }

    var tracing: Boolean = false
}

var CpuState._thread: PspThread? by Extra.Property { null }
val CpuState.thread: PspThread get() = _thread ?: invalidOp1("CpuState doesn't have a thread attached")

data class PspEventFlag(override val id: Int) : ResourceItem {
    var name: String = ""
    var attributes: Int = 0
    var currentPattern: Int = 0
    var optionsPtr: Ptr? = null

    fun poll(bitsToMatch: Int, waitType: Int, outBits: Ptr): Boolean {
        if (outBits.isNotNull) outBits.sw(0, this.currentPattern)

        val res = when {
            (waitType and EventFlagWaitTypeSet.Or) != 0 -> ((this.currentPattern and bitsToMatch) != 0) // one or more bits of the mask
            else -> (this.currentPattern and bitsToMatch) == bitsToMatch // all the bits of the mask
        }

        if (res) {
            this._doClear(bitsToMatch, waitType)
            return true
        } else {
            return false
        }
    }

    private fun _doClear(bitsToMatch: Int, waitType: Int) {
        if ((waitType and (EventFlagWaitTypeSet.ClearAll)) != 0) this.clearBits(-1.inv(), false)
        if ((waitType and (EventFlagWaitTypeSet.Clear)) != 0) this.clearBits(bitsToMatch.inv(), false)
    }

    fun clearBits(bitsToClear: Int, doUpdateWaitingThreads: Boolean = true) {
        this.currentPattern = this.currentPattern and bitsToClear
        if (doUpdateWaitingThreads) this.updateWaitingThreads()
    }

    private fun updateWaitingThreads() {
        //this.waitingThreads.forEach(waitingThread => {
        //	if (this.poll(waitingThread.bitsToMatch, waitingThread.waitType, waitingThread.outBits)) {
        //		waitingThread.wakeUp();
        //	}
        //});
    }

    fun setBits(bits: Int, doUpdateWaitingThreads: Boolean = true) {
        this.currentPattern = this.currentPattern or bits
        if (doUpdateWaitingThreads) this.updateWaitingThreads()
    }
}

object EventFlagWaitTypeSet {
    const val And = 0x00
    const val Or = 0x01
    const val ClearAll = 0x10
    const val Clear = 0x20
    const val MaskValidBits = Or or Clear or ClearAll
}

object ThreadStatus {
    const val RUNNING = 1
    const val READY = 2
    const val WAIT = 4
    const val SUSPEND = 8
    const val DORMANT = 16
    const val DEAD = 32
    const val WAITSUSPEND = WAIT or SUSPEND
}

class SceKernelThreadInfo(
    var size: Int = 0,
    var name: String = "",
    var attributes: Int = 0,
    var status: Int = 0, // ThreadStatus
    var entryPoint: Int = 0,
    var stackPointer: Int = 0,
    var stackSize: Int = 0,
    var GP: Int = 0,
    var priorityInit: Int = 0,
    var priority: Int = 0,
    var waitType: Int = 0,
    var waitId: Int = 0,
    var wakeupCount: Int = 0,
    var exitStatus: Int = 0,
    var runClocksLow: Int = 0,
    var runClocksHigh: Int = 0,
    var interruptPreemptionCount: Int = 0,
    var threadPreemptionCount: Int = 0,
    var releaseCount: Int = 0
) {
    companion object : Struct<SceKernelThreadInfo>(
        { SceKernelThreadInfo() },
        SceKernelThreadInfo::size AS INT32,
        SceKernelThreadInfo::name AS STRINGZ(32),
        SceKernelThreadInfo::attributes AS INT32,
        SceKernelThreadInfo::status AS INT32,
        SceKernelThreadInfo::entryPoint AS INT32,
        SceKernelThreadInfo::stackPointer AS INT32,
        SceKernelThreadInfo::stackSize AS INT32,
        SceKernelThreadInfo::GP AS INT32,
        SceKernelThreadInfo::priorityInit AS INT32,
        SceKernelThreadInfo::priority AS INT32,
        SceKernelThreadInfo::waitType AS INT32,
        SceKernelThreadInfo::waitId AS INT32,
        SceKernelThreadInfo::wakeupCount AS INT32,
        SceKernelThreadInfo::exitStatus AS INT32,
        SceKernelThreadInfo::runClocksLow AS INT32,
        SceKernelThreadInfo::runClocksHigh AS INT32,
        SceKernelThreadInfo::interruptPreemptionCount AS INT32,
        SceKernelThreadInfo::threadPreemptionCount AS INT32,
        SceKernelThreadInfo::releaseCount AS INT32
    )
}

object PspThreadAttributes {
    const val None = 0
    const val LowFF = 0x000000FF.toInt()
    const val Vfpu = 0x00004000.toInt() // Enable VFPU access for the thread.
    const val V0x2000 = 0x2000.toInt()
    const val V0x4000 = 0x4000.toInt()
    const val V0x400000 = 0x400000.toInt()
    const val V0x800000 = 0x800000.toInt()
    const val V0xf00000 = 0xf00000.toInt()
    const val V0x8000000 = 0x8000000.toInt()
    const val V0xf000000 = 0xf000000.toInt()
    const val User = 0x80000000.toInt() // Start the thread in user mode (done automatically if the thread creating it is in user mode).
    const val UsbWlan = 0xa0000000.toInt() // Thread is part of the USB/WLAN API.
    const val Vsh = 0xc0000000.toInt() // Thread is part of the VSH API.
    //val ScratchRamEnable = 0x00008000, // Allow using scratchpad memory for a thread, NOT USABLE ON V1.0
    const val NoFillStack = 0x00100000.toInt() // Disables filling the stack with 0xFF on creation
    const val ClearStack = 0x00200000.toInt() // Clear the stack when the thread is deleted
    const val ValidMask = (LowFF or Vfpu or User or UsbWlan or Vsh or /*ScratchRamEnable |*/ NoFillStack or ClearStack or V0x2000 or V0x4000 or V0x400000 or V0x800000 or V0xf00000 or V0x8000000 or V0xf000000).toInt()
}