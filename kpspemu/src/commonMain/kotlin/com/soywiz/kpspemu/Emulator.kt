package com.soywiz.kpspemu

import com.soywiz.klogger.*
import com.soywiz.korau.format.util.*
import com.soywiz.korinject.*
import com.soywiz.korio.async.*
import com.soywiz.kpspemu.battery.*
import com.soywiz.kpspemu.cpu.*
import com.soywiz.kpspemu.cpu.dis.*
import com.soywiz.kpspemu.ctrl.*
import com.soywiz.kpspemu.display.*
import com.soywiz.kpspemu.ge.*
import com.soywiz.kpspemu.hle.manager.*
import com.soywiz.kpspemu.mem.*
import kotlin.coroutines.*

class Emulator constructor(
    val coroutineContext: CoroutineContext,
    val syscalls: SyscallManager = SyscallManager(),
    val mem: Memory = Memory(),
    var gpuRenderer: GpuRenderer = DummyGpuRenderer()
) : AsyncDependency {
    //val INITIAL_INTERPRETED = false
    val INITIAL_INTERPRETED = true

    var interpreted = INITIAL_INTERPRETED

    val onHomePress = Signal<Unit>()
    val onLoadPress = Signal<Unit>()
    val logger = Logger("Emulator")
    val timeManager = TimeManager(this)
    val nameProvider = AddressInfo()
    val breakpoints = Breakpoints()
    val globalCpuState = GlobalCpuState(mem)
    var output = StringBuilder()
    val ge: Ge = Ge(this)
    val gpu: Gpu = Gpu(this)
    val battery: PspBattery = PspBattery(this)
    val interruptManager: InterruptManager = InterruptManager(this)
    val display: PspDisplay = PspDisplay(this)
    val deviceManager = DeviceManager(this)
    val configManager = ConfigManager()
    val memoryManager = MemoryManager(this)
    val threadManager = ThreadManager(this)
    val moduleManager = ModuleManager(this)
    val callbackManager = CallbackManager(this)
    val controller = PspController(this)
    val fileManager = FileManager(this)
    val imem = object : IMemory {
        override fun read8(addr: Int): Int = mem.lbu(addr)
    }

    override suspend fun init() {
        configManager.init()
        deviceManager.init()

        configManager.storage.subscribe {
            deviceManager.setStorage(it)
        }
    }

    val running: Boolean get() = threadManager.aliveThreadCount >= 1
    var globalTrace: Boolean = false
    var sdkVersion: Int = 150

    init {
        CpuBreakException.initialize(mem)
    }

    fun invalidateInstructionCache(ptr: Int = 0, size: Int = Int.MAX_VALUE) {
        logger.trace { "invalidateInstructionCache($ptr, $size)" }
        globalCpuState.mcache.invalidateInstructionCache(ptr, size)
    }

    fun dataCache(ptr: Int = 0, size: Int = Int.MAX_VALUE, writeback: Boolean, invalidate: Boolean) {
        logger.trace { "writebackDataCache($ptr, $size, writeback=$writeback, invalidate=$invalidate)" }
    }

    suspend fun reset() {
        globalTrace = false
        sdkVersion = 150
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
