package com.soywiz.kpspemu.hle.manager

import com.soywiz.korio.async.Promise
import com.soywiz.korio.async.Signal
import com.soywiz.korio.error.invalidOp
import com.soywiz.korio.util.Extra
import com.soywiz.korio.util.nextAlignedTo
import com.soywiz.kpspemu.*
import com.soywiz.kpspemu.cpu.CpuBreakException
import com.soywiz.kpspemu.cpu.CpuState
import com.soywiz.kpspemu.cpu.RA
import com.soywiz.kpspemu.cpu.SP
import com.soywiz.kpspemu.cpu.interpreter.CpuInterpreter
import com.soywiz.kpspemu.mem.Ptr
import com.soywiz.kpspemu.mem.PtrArray
import com.soywiz.kpspemu.mem.array
import com.soywiz.kpspemu.mem.ptr
import com.soywiz.kpspemu.util.PspLogger

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

	fun vblank() {
		for (t in resourcesById.values.filter { it.waitObject is WaitObject.VBLANK }) {
			t.resume()
			//println("RESUMED WAITING THREAD!")
		}
	}

	fun step() {
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
	val attributes: Int,
	val optionPtr: Ptr
) : Resource(threadManager, id, name), WithEmulator {
	val totalExecutedInstructions: Long get() = state.totalExecuted
	val onEnd = Signal<Unit>()
	val logger = PspLogger("PspThread")

	enum class Phase { STOPPED, RUNNING, WAITING, DELETED }

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
		manager.freeIds.free(id)
		manager.resourcesById.remove(id)
	}

	fun exitAndKill() {
		stop()
		delete()
	}

	fun step(now: Double) {
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