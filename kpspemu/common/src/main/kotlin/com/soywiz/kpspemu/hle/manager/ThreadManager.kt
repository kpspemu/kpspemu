package com.soywiz.kpspemu.hle.manager

import com.soywiz.korio.error.invalidOp
import com.soywiz.korio.util.Extra
import com.soywiz.korio.util.Pool
import com.soywiz.kpspemu.Emulator
import com.soywiz.kpspemu.WithEmulator
import com.soywiz.kpspemu.cpu.CpuBreak
import com.soywiz.kpspemu.cpu.CpuState
import com.soywiz.kpspemu.cpu.RA
import com.soywiz.kpspemu.cpu.SP
import com.soywiz.kpspemu.cpu.interpreter.CpuInterpreter
import com.soywiz.kpspemu.mem
import com.soywiz.kpspemu.mem.Ptr
import com.soywiz.kpspemu.mem.ptr

class ThreadManager(val emulator: Emulator) {
	val memoryManager get() = emulator.memoryManager
	var lastId: Int = 0
	val freeIds = Pool { lastId++ }
	val threadsById = LinkedHashMap<Int, PspThread>()

	fun createThread(name: String, entryPoint: Int, initPriority: Int, stackSize: Int, attributes: Int, optionPtr: Ptr): PspThread {
		val stack = memoryManager.userPartition.allocateHigh(stackSize, "${name}_stack")
		val thread = PspThread(this, freeIds.alloc(), name, entryPoint, stack, initPriority, attributes, optionPtr)
		threadsById[thread.id] = thread
		return thread
	}

	fun step() {
		val availableThreads = threadsById.values.filter { it.running }.sortedBy { it.priority }
		for (t in availableThreads) {
			t.step()
		}
	}
}

class PspThreadQueue {

}

class PspThread internal constructor(
	val manager: ThreadManager,
	val id: Int,
	val name: String,
	val entryPoint: Int,
	val stack: MemoryPartition,
	val initPriority: Int,
	val attributes: Int,
	val optionPtr: Ptr
) : WithEmulator {
	enum class Phase { Stopped, Running, Waiting, Deleted, }

	var phase: Phase = Phase.Stopped
	val running: Boolean get() = phase == Phase.Running
	var priority: Int = initPriority
	override val emulator get() = manager.emulator
	val state = CpuState(emulator.mem, emulator.syscalls).apply {
		_thread = this@PspThread
		setPC(entryPoint)
		SP = stack.high.toInt()
	}
	val interpreter = CpuInterpreter(state)

	init {
		//val ptr = putWordInStack(0b000000_00000000000000000000_001101 or (77 shl 6)) // break 77
		state.RA = putWordsInStack(
			0b000000_00000000000000000000_001101 or (CpuBreak.THREAD_EXIT_KILL shl 6), // break 77
			0b000000_00000000000000000000_000000, // nop
			0b000000_00000000000000000000_000000, // nop
			0b000000_00000000000000000000_000000  // nop
		).addr
	}

	fun putDataInStack(bytes: ByteArray): Ptr {
		state.SP -= bytes.size
		mem.write(state.SP, bytes)
		return mem.ptr(state.SP)
	}

	fun putWordInStack(word: Int): Ptr {
		state.SP -= 4
		mem.sw(state.SP, word)
		return mem.ptr(state.SP)
	}

	fun putWordsInStack(vararg words: Int): Ptr {
		state.SP -= words.size * 4
		for (n in 0 until words.size) mem.sw(state.SP + n * 4, words[n])
		return mem.ptr(state.SP)
	}

	fun start() {
		phase = Phase.Running
	}

	fun stop() {
		phase = Phase.Stopped
	}

	fun delete() {
		phase = Phase.Deleted
		manager.freeIds.free(id)
		manager.threadsById.remove(id)
	}

	fun exitAndKill() {
		stop()
		delete()
	}

	fun step() {
		try {
			interpreter.steps(1_000_000)
		} catch (e: CpuBreak) {
			when (e.id) {
				CpuBreak.THREAD_EXIT_KILL -> {
					println("BREAK: THREAD_EXIT_KILL")
					exitAndKill()
				}
				else -> throw e
			}
		}
	}
}

var CpuState._thread: PspThread? by Extra.Property { null }
val CpuState.thread: PspThread get() = _thread ?: invalidOp("CpuState doesn't have a thread attached")