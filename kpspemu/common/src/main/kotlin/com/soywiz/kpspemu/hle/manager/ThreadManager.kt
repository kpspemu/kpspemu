package com.soywiz.kpspemu.hle.manager

import com.soywiz.kds.Extra
import com.soywiz.klogger.Logger
import com.soywiz.korio.async.Promise
import com.soywiz.korio.async.Signal
import com.soywiz.korio.error.invalidOp
import com.soywiz.korio.util.nextAlignedTo
import com.soywiz.kpspemu.*
import com.soywiz.kpspemu.cpu.CpuBreakException
import com.soywiz.kpspemu.cpu.CpuState
import com.soywiz.kpspemu.cpu.RA
import com.soywiz.kpspemu.cpu.SP
import com.soywiz.kpspemu.cpu.interpreter.CpuInterpreter
import com.soywiz.kpspemu.mem.*
import com.soywiz.kpspemu.util.*

//const val INSTRUCTIONS_PER_STEP = 500_000
//const val INSTRUCTIONS_PER_STEP = 1_000_000
//const val INSTRUCTIONS_PER_STEP = 2_000_000
//const val INSTRUCTIONS_PER_STEP = 4_000_000
const val INSTRUCTIONS_PER_STEP = 5_000_000
//const val INSTRUCTIONS_PER_STEP = 10_000_000
//const val INSTRUCTIONS_PER_STEP = 100_000_000

class ThreadManager(emulator: Emulator) : Manager<PspThread>("Thread", emulator) {
	val threads get() = resourcesById.values
	val waitingThreads: Int get() = resourcesById.count { it.value.waiting }
	val activeThreads: Int get() = resourcesById.count { it.value.running }
	val totalThreads: Int get() = resourcesById.size
	val aliveThreadCount: Int get() = resourcesById.values.count { it.running || it.waiting }

	fun create(name: String, entryPoint: Int, initPriority: Int, stackSize: Int, attributes: Int, optionPtr: Ptr): PspThread {
		val stack = memoryManager.userPartition.allocateHigh(stackSize, "${name}_stack")
		return PspThread(this, allocId(), name, entryPoint, stack, initPriority, attributes, optionPtr)
	}

	fun suspend() {
		throw CpuBreakException(CpuBreakException.THREAD_WAIT)
	}

	fun step() {
		var startTime = rtc.getTimeInMicrosecondsDouble()
		var vsyncStarted = true

		display.startVsync()
		emulator.interruptManager.dispatchVsync()
		for (t in resourcesById.values.filter { it.waitObject is WaitObject.VBLANK }) {
			t.resume()
			//println("RESUMED WAITING THREAD!")
		}

		while ((rtc.getTimeInMicrosecondsDouble() - startTime) < 16.0) { // Max 16 milliseconds
			stepOne()
			emulator.eventLoop.step(0)

			// This is trick to simulate vsync start
			if (vsyncStarted) {
				display.endVsync();
				vsyncStarted = false
			}
		}
	}

	fun stepOne() {
		val now: Double = timeManager.getTimeInMicrosecondsDouble()

		for (t in resourcesById.values.filter { it.waitObject is WaitObject.TIME }) {
			val time = (t.waitObject as WaitObject.TIME).instant
			if (now >= time) {
				t.resume()
			}
		}

		val availableThreads = resourcesById.values.filter { it.running }.sortedBy { it.priority }
		for (t in availableThreads) {
			t.step(now)
			if (availableThreads.isEmpty()) break
		}
	}

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
		try {
			cpu.setPC(address)
			cpu.RA = CpuBreakException.INTERRUPT_RETURN_RA
			cpu.r4 = argument
			mem.sw(CpuBreakException.INTERRUPT_RETURN_RA, 0b000000_00000000000000000000_001101 or (CpuBreakException.INTERRUPT_RETURN shl 6))

			thread.step(timeManager.getTimeInMicrosecondsDouble())
		} catch (e: CpuBreakException) {
			// END OF INTERRUPT
		} finally {
			cpu.setTo(backCpu)
			gcpustate.insideInterrupt = oldInsideInterrupt
		}
	}

	fun delayThread(micros: Int) {
		// @TODO:
	}
}

sealed class WaitObject {
	data class TIME(val instant: Double) : WaitObject()
	data class PROMISE(val promise: Promise<Unit>, val reason: String) : WaitObject()
	object SLEEP : WaitObject()
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
	val onEnd = Signal<Unit>()
	val logger = Logger("PspThread")

	enum class Phase {
		STOPPED,
		RUNNING,
		WAITING,
		DELETED
	}

	val status: Int get() {
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
	val state = CpuState(emulator.globalCpuState, emulator.mem, emulator.syscalls).apply {
		_thread = this@PspThread
		setPC(entryPoint)
		SP = stack.high.toInt()
	}
	val interpreter = CpuInterpreter(state)
	//val interpreter = FastCpuInterpreter(state)

	init {
		updateTrace()
	}

	fun updateTrace() {
		interpreter.trace = threadManager.traces[name] == true
	}

	init {
		//val ptr = putWordInStack(0b000000_00000000000000000000_001101 or (77 shl 6)) // break 77
		state.RA = putWordsInStack(
			0b000000_00000000000000000000_001101 or (CpuBreakException.THREAD_EXIT_KILL shl 6), // break 77
			0b000000_00000000000000000000_000000, // nop
			0b000000_00000000000000000000_000000, // nop
			0b000000_00000000000000000000_000000  // nop
		).addr
	}

	fun putDataInStack(bytes: ByteArray): PtrArray {
		val blockSize = bytes.size.nextAlignedTo(16)
		state.SP -= blockSize
		mem.write(state.SP, bytes)
		return PtrArray(mem.ptr(state.SP), bytes.size)
	}

	fun putWordInStack(word: Int): PtrArray {
		val blockSize = 4.nextAlignedTo(16)
		state.SP -= blockSize
		mem.sw(state.SP, word)
		return mem.ptr(state.SP).array(4)
	}

	fun putWordsInStack(vararg words: Int): PtrArray {
		val blockSize = (words.size * 4).nextAlignedTo(16)
		state.SP -= blockSize
		for (n in 0 until words.size) mem.sw(state.SP + n * 4, words[n])
		return mem.ptr(state.SP).array(words.size * 4)
	}

	fun start() {
		resume()
	}

	fun resume() {
		phase = Phase.RUNNING
		waitObject = null
		waitInfo = null
		acceptingCallbacks = false
	}

	fun stop(reason: String = "generic") {
		if (phase != Phase.STOPPED) {
			phase = Phase.STOPPED
			onEnd(Unit)
		}
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

	fun step(now: Double) {
		preemptionCount++
		try {
			interpreter.steps(INSTRUCTIONS_PER_STEP)
		} catch (e: CpuBreakException) {
			when (e.id) {
				CpuBreakException.THREAD_EXIT_KILL -> {
					logger.info("BREAK: THREAD_EXIT_KILL ('${this.name}', ${this.id})")
					exitAndKill()
				}
				CpuBreakException.THREAD_WAIT -> {
				}
				else -> throw e
			}
		}
	}

	fun markWaiting(wait: WaitObject, cb: Boolean) {
		this.waitObject = wait
		this.phase = Phase.WAITING
		this.acceptingCallbacks = cb
	}

	fun suspend(wait: WaitObject, cb: Boolean) {
		markWaiting(wait, cb)
		if (wait is WaitObject.PROMISE) {
			wait.promise.then { resume() }
		}
		threadManager.suspend()
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
		if ((waitType and (EventFlagWaitTypeSet.ClearAll)) != 0) this.clearBits(-1.inv(), false);
		if ((waitType and (EventFlagWaitTypeSet.Clear)) != 0) this.clearBits(bitsToMatch.inv(), false);
	}

	fun clearBits(bitsToClear: Int, doUpdateWaitingThreads: Boolean = true) {
		this.currentPattern = this.currentPattern and bitsToClear;
		if (doUpdateWaitingThreads) this.updateWaitingThreads();
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

