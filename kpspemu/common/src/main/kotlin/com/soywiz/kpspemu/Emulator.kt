package com.soywiz.kpspemu

import com.soywiz.kpspemu.battery.PspBattery
import com.soywiz.kpspemu.cpu.GlobalCpuState
import com.soywiz.kpspemu.ctrl.PspController
import com.soywiz.kpspemu.display.PspDisplay
import com.soywiz.kpspemu.ge.DummyGpuRenderer
import com.soywiz.kpspemu.ge.Ge
import com.soywiz.kpspemu.ge.Gpu
import com.soywiz.kpspemu.ge.GpuRenderer
import com.soywiz.kpspemu.hle.manager.*
import com.soywiz.kpspemu.mem.Memory
import com.soywiz.kpspemu.util.hex
import kotlin.coroutines.experimental.CoroutineContext

class Emulator(
	val coroutineContext: CoroutineContext,
	val syscalls: SyscallManager = SyscallManager(),
	val mem: Memory = Memory(),
	val gpuRenderer: GpuRenderer = DummyGpuRenderer()
) {
	val globalCpuState = GlobalCpuState()
	val logger = com.soywiz.kpspemu.util.PspLogger("Emulator")

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
	val timeManager = TimeManager(this)
	val controller = PspController(this)
	val fileManager = FileManager(this)

	val running: Boolean get() = threadManager.aliveThreadCount >= 1

	fun frameStep() {
		controller.startFrame(timeManager.getTimeInMicrosecondsDouble().toInt())
		threadManager.vblank()
		display.dispatchVsync()
		threadManager.step()
		ge.run()
		gpu.render()
		controller.endFrame()
	}

	fun invalidateIcache(ptr: Int, size: Int) {
		logger.trace { "invalidateIcache(${ptr.hex}, $size)" }
	}

	fun invalidateDcache(ptr: Int, size: Int) {
		logger.trace { "invalidateDcache()" }
	}
}

interface WithEmulator {
	val emulator: Emulator
}

val WithEmulator.mem: Memory get() = emulator.mem
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
