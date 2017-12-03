package com.soywiz.kpspemu.hle.manager

import com.soywiz.kds.Extra
import com.soywiz.klogger.Logger
import com.soywiz.korio.async.sleep
import com.soywiz.korio.error.invalidOp
import com.soywiz.korio.lang.format
import com.soywiz.korio.lang.printStackTrace
import com.soywiz.korio.util.hasFlag
import com.soywiz.korio.util.hex
import com.soywiz.korio.util.nextAlignedTo
import com.soywiz.korio.util.umod
import com.soywiz.kpspemu.*
import com.soywiz.kpspemu.cpu.*
import com.soywiz.kpspemu.cpu.interpreter.CpuInterpreter
import com.soywiz.kpspemu.mem.*
import com.soywiz.kpspemu.util.*
import kotlin.math.max

//const val INSTRUCTIONS_PER_STEP = 500_000
//const val INSTRUCTIONS_PER_STEP = 1_000_000
//const val INSTRUCTIONS_PER_STEP = 2_000_000
//const val INSTRUCTIONS_PER_STEP = 4_000_000
const val INSTRUCTIONS_PER_STEP = 5_000_000
//const val INSTRUCTIONS_PER_STEP = 10_000_000
//const val INSTRUCTIONS_PER_STEP = 100_000_000


class ThreadManager(emulator: Emulator) : Manager<PspThread>("Thread", emulator) {
	val logger = Logger("ThreadManager").apply {
		//level = LogLevel.TRACE
	}
	val threads get() = resourcesById.values
	val waitingThreads: Int get() = resourcesById.count { it.value.waiting }
	val activeThreads: Int get() = resourcesById.count { it.value.running }
	val totalThreads: Int get() = resourcesById.size
	val aliveThreadCount: Int get() = resourcesById.values.count { it.running || it.waiting }
	val onThreadChanged = Signal2<PspThread>()
	var currentThread: PspThread? = null

	override fun reset() {
		super.reset()
		currentThread = null
	}

	fun create(name: String, entryPoint: Int, initPriority: Int, stackSize: Int, attributes: Int, optionPtr: Ptr): PspThread {
		//priorities.sortBy { it.priority }
		var attr = attributes
		val ssize = max(stackSize, 0x200).nextAlignedTo(0x100)
		//val ssize = max(stackSize, 0x20000).nextAlignedTo(0x10000)
		val stack = memoryManager.userPartition.allocateHigh(ssize, "${name}_stack")

		println(stack.toString2())

		val thread = PspThread(this, allocId(), name, entryPoint, stack, initPriority, attributes, optionPtr)
		logger.info { "stack:%08X-%08X (%d)".format(stack.low.toInt(), stack.high.toInt(), stack.size.toInt()) }

		memoryManager.userPartition.dump()

		attr = attr or PspThreadAttributes.User
		attr = attr or PspThreadAttributes.LowFF

		if (!(attr hasFlag PspThreadAttributes.NoFillStack)) {
			logger.trace { "FILLING: $stack" }
			mem.fill(-1, stack.low.toInt(), stack.size.toInt())
		} else {
			logger.trace { "NOT FILLING: $stack" }
		}
		threadManager.onThreadChanged(thread)
		return thread
	}

	fun suspend() {
		throw CpuBreakException(CpuBreakException.THREAD_WAIT)
	}

	fun getActiveThreadPriorities(): List<Int> = threads.filter { it.running }.map { it.priority }.distinct().sorted()
	fun getActiveThreadsWithPriority(priority: Int): List<PspThread> = threads.filter { it.running && it.priority == priority }
	fun getFirstThread(): PspThread? = getActiveThreadPriorities().firstOrNull()?.let { getActiveThreadsWithPriority(it) }?.firstOrNull()

	fun computeNextThread(prevThread: PspThread?): PspThread? {
		if (prevThread == null) return getFirstThread()
		val threadsWithPriority = getActiveThreadsWithPriority(prevThread.priority)
		val threadsWithPriorityCount = threadsWithPriority.size
		if (threadsWithPriorityCount == 0) return null
		val index = threadsWithPriority.indexOf(prevThread) umod threadsWithPriorityCount
		return threadsWithPriority.getOrNull((index + 1) umod threadsWithPriorityCount)
	}

	suspend fun waitThreadChange() {
		//println("[1]")
		try {
			onThreadChanged.waitOne(16)
		} catch (e: TimeoutException) {
		}
		//println("[2]")
		//coroutineContext.sleep(0)
	}

	suspend fun step() {
		val start: Double = timeManager.getTimeInMicrosecondsDouble()

		do {
			val now: Double = timeManager.getTimeInMicrosecondsDouble()
			if (currentThread == null) currentThread = getFirstThread()
			if (currentThread?.running == false) {
				println("WAIT! Trying to execute a sleeping thread!")
				currentThread = computeNextThread(currentThread)
			}
			try {
				currentThread?.step(now)
				currentThread = computeNextThread(currentThread)
			} catch (e: BreakpointException) {
				break
			}
			if (now - start >= 16.0) {
				break // Rest a bit
			}
		} while (currentThread != null)
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
		throw CpuBreakException(CpuBreakException.THREAD_EXIT_KILL)
	}

	fun executeInterrupt(address: Int, argument: Int) {
		val gcpustate = emulator.globalCpuState
		val oldInsideInterrupt = gcpustate.insideInterrupt
		gcpustate.insideInterrupt = true
		val thread = threads.first()
		val cpu = thread.state
		val backCpu = cpu.clone()

		cpu.setPC(address)
		cpu.RA = CpuBreakException.INTERRUPT_RETURN_RA
		cpu.r4 = argument

		while (true) {
			val res = thread.step(timeManager.getTimeInMicrosecondsDouble(), trace = false)
			if (res != 0) break
		}
		cpu.setTo(backCpu)
		gcpustate.insideInterrupt = oldInsideInterrupt
	}

	fun delayThread(micros: Int) {
		// @TODO:
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
	var preemptionCount: Int = 0
	val totalExecutedInstructions: Long get() = state.totalExecuted
	val onEnd = Signal2<Unit>()
	val onWakeUp = Signal2<Unit>()
	val logger = Logger("PspThread")

	enum class Phase {
		STOPPED,
		RUNNING,
		WAITING,
		DELETED
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
	val running: Boolean get() = phase == Phase.RUNNING
	val waiting: Boolean get() = waitObject != null
	var priority: Int = initPriority
	override val emulator get() = manager.emulator
	val state = CpuState("state.thread.$name", emulator.globalCpuState, emulator.mem, emulator.syscalls).apply {
		_thread = this@PspThread
		setPC(entryPoint)
		SP = stack.high.toInt()
		RA = CpuBreakException.THREAD_EXIT_KIL_RA

		println("CREATED THREAD('$name'): PC=${PC.hex}, SP=${SP.hex}")
	}
	val interpreter = CpuInterpreter(state, emulator.breakpoints, emulator.nameProvider)
	//val interpreter = FastCpuInterpreter(state)

	init {
		updateTrace()
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
		threadManager.onThreadChanged(this)
	}

	fun resume() {
		phase = Phase.RUNNING
		waitObject = null
		waitInfo = null
		acceptingCallbacks = false
		threadManager.onThreadChanged(this)
	}

	fun stop(reason: String = "generic") {
		if (phase != Phase.STOPPED) {
			phase = Phase.STOPPED
			onEnd(Unit)
		}
		//threadManager.onThreadChanged(this)
	}

	fun delete() {
		stop()
		phase = Phase.DELETED
		manager.freeById(id)
	}

	fun exitAndKill() {
		stop()
		delete()
	}

	fun step(now: Double, trace: Boolean = false): Int {
		//if (name == "update_thread") {
		//	println("Ignoring: Thread.${this.name}")
		//	stop("ignoring")
		//	return
		//}
		//println("Step: Thread.${this.name}")
		preemptionCount++
		try {
			interpreter.steps(INSTRUCTIONS_PER_STEP, trace)
			return 0
		} catch (e: CpuBreakException) {
			when (e.id) {
				CpuBreakException.THREAD_EXIT_KILL -> {
					logger.info { "BREAK: THREAD_EXIT_KILL ('${this.name}', ${this.id})" }
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
					e.printStackTrace()
					throw e
				}
			}
			return e.id
		}
	}

	fun markWaiting(wait: WaitObject, cb: Boolean) {
		this.waitObject = wait
		this.phase = Phase.WAITING
		this.acceptingCallbacks = cb
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
		coroutineContext.sleep(totalMicroseconds / 1000)
	}

	suspend fun sleepSeconds(seconds: Double) = sleepMicro((seconds * 1_000_000).toInt())

	suspend fun sleepSecondsIfRequired(seconds: Double) {
		if (seconds > 0.0) sleepMicro((seconds * 1_000_000).toInt())
	}
}

var CpuState._thread: PspThread? by Extra.Property { null }
val CpuState.thread: PspThread get() = _thread ?: invalidOp("CpuState doesn't have a thread attached")

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
	val And = 0x00
	val Or = 0x01
	val ClearAll = 0x10
	val Clear = 0x20
	val MaskValidBits = Or or Clear or ClearAll
}

object ThreadStatus {
	val RUNNING = 1
	val READY = 2
	val WAIT = 4
	val SUSPEND = 8
	val DORMANT = 16
	val DEAD = 32
	val WAITSUSPEND = WAIT or SUSPEND
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
	companion object : Struct<SceKernelThreadInfo>({ SceKernelThreadInfo() },
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
	val None = 0
	val LowFF = 0x000000FF.toInt()
	val Vfpu = 0x00004000.toInt() // Enable VFPU access for the thread.
	val V0x2000 = 0x2000.toInt()
	val V0x4000 = 0x4000.toInt()
	val V0x400000 = 0x400000.toInt()
	val V0x800000 = 0x800000.toInt()
	val V0xf00000 = 0xf00000.toInt()
	val V0x8000000 = 0x8000000.toInt()
	val V0xf000000 = 0xf000000.toInt()
	val User = 0x80000000.toInt() // Start the thread in user mode (done automatically if the thread creating it is in user mode).
	val UsbWlan = 0xa0000000.toInt() // Thread is part of the USB/WLAN API.
	val Vsh = 0xc0000000.toInt() // Thread is part of the VSH API.
	//val ScratchRamEnable = 0x00008000, // Allow using scratchpad memory for a thread, NOT USABLE ON V1.0
	val NoFillStack = 0x00100000.toInt() // Disables filling the stack with 0xFF on creation
	val ClearStack = 0x00200000.toInt() // Clear the stack when the thread is deleted
	val ValidMask = (LowFF or Vfpu or User or UsbWlan or Vsh or /*ScratchRamEnable |*/ NoFillStack or ClearStack or V0x2000 or V0x4000 or V0x400000 or V0x800000 or V0xf00000 or V0x8000000 or V0xf000000).toInt()
}