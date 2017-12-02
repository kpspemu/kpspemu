package com.soywiz.kpspemu

import com.soywiz.klogger.Logger
import com.soywiz.korau.format.util.IMemory
import com.soywiz.korio.async.eventLoop
import com.soywiz.korio.util.hex
import com.soywiz.kpspemu.battery.PspBattery
import com.soywiz.kpspemu.cpu.Breakpoints
import com.soywiz.kpspemu.cpu.CpuBreakException
import com.soywiz.kpspemu.cpu.GlobalCpuState
import com.soywiz.kpspemu.cpu.dis.NameProvider
import com.soywiz.kpspemu.ctrl.PspController
import com.soywiz.kpspemu.display.PspDisplay
import com.soywiz.kpspemu.ge.DummyGpuRenderer
import com.soywiz.kpspemu.ge.Ge
import com.soywiz.kpspemu.ge.Gpu
import com.soywiz.kpspemu.ge.GpuRenderer
import com.soywiz.kpspemu.hle.manager.*
import com.soywiz.kpspemu.mem.Memory
import kotlin.coroutines.experimental.CoroutineContext

class Emulator(
	val coroutineContext: CoroutineContext,
	val syscalls: SyscallManager = SyscallManager(),
	val mem: Memory = Memory(),
	var gpuRenderer: GpuRenderer = DummyGpuRenderer()
) {
	val logger = Logger("Emulator")
	val timeManager = TimeManager(this)
	val nameProvider = AddressInfo()
	val breakpoints = Breakpoints()
	val globalCpuState = GlobalCpuState()
	var output = StringBuilder()
	val ge: Ge = Ge(this)
	val gpu: Gpu = Gpu(this)
	val battery: PspBattery = PspBattery(this)
	val interruptManager: InterruptManager = InterruptManager(this)
	val display: PspDisplay = PspDisplay(this)
	val deviceManager = DeviceManager(this)
	val memoryManager = MemoryManager(this)
	val threadManager = ThreadManager(this)
	val moduleManager = ModuleManager(this)
	val callbackManager = CallbackManager(this)
	val controller = PspController(this)
	val fileManager = FileManager(this)
	val imem = object : IMemory {
		override fun read8(addr: Int): Int = mem.lbu(addr)
	}

	val running: Boolean get() = threadManager.aliveThreadCount >= 1

	init {
		CpuBreakException.initialize(mem)
	}

	fun invalidateIcache(ptr: Int, size: Int) {
		logger.trace { "invalidateIcache(${ptr.hex}, $size)" }
	}

	fun invalidateDcache(ptr: Int, size: Int) {
		logger.trace { "invalidateDcache()" }
	}

	fun reset() {
		syscalls.reset()
		mem.reset()
		CpuBreakException.initialize(mem)
		gpuRenderer.reset()
		timeManager.reset()
		nameProvider.reset()
		//breakpoints.reset() // Do not reset breakpoints?
		globalCpuState.reset()
		output = StringBuilder()
		ge.reset()
		gpu.reset()
		battery.reset()
		interruptManager.reset()
		display.reset()
		deviceManager.reset()
		memoryManager.reset()
		threadManager.reset()
		moduleManager.reset()
		callbackManager.reset()
		controller.reset()
		fileManager.reset()
	}
}

interface WithEmulator {
	val emulator: Emulator
}

class AddressInfo : NameProvider {
	val names = hashMapOf<Int, String>()
	override fun getName(addr: Int): String? = names[addr]
	fun reset() {
		names.clear()
	}
}

val WithEmulator.mem: Memory get() = emulator.mem
val WithEmulator.imem: IMemory get() = emulator.imem
val WithEmulator.ge: Ge get() = emulator.ge
val WithEmulator.gpu: Gpu get() = emulator.gpu
val WithEmulator.controller: PspController get() = emulator.controller
val WithEmulator.coroutineContext: CoroutineContext get() = emulator.coroutineContext
val WithEmulator.display: PspDisplay get() = emulator.display
val WithEmulator.deviceManager: DeviceManager get() = emulator.deviceManager
val WithEmulator.memoryManager: MemoryManager get() = emulator.memoryManager
val WithEmulator.timeManager: TimeManager get() = emulator.timeManager
val WithEmulator.fileManager: FileManager get() = emulator.fileManager
val WithEmulator.rtc: TimeManager get() = emulator.timeManager
val WithEmulator.threadManager: ThreadManager get() = emulator.threadManager
val WithEmulator.callbackManager: CallbackManager get() = emulator.callbackManager
val WithEmulator.breakpoints: Breakpoints get() = emulator.breakpoints
