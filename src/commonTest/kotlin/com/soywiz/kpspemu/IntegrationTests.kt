package com.soywiz.kpspemu

import com.soywiz.klock.*
import com.soywiz.klogger.*
import com.soywiz.korio.async.*
import com.soywiz.korio.file.std.*
import com.soywiz.korio.lang.*
import com.soywiz.korio.stream.*
import com.soywiz.korio.util.*
import com.soywiz.kpspemu.format.elf.*
import com.soywiz.kpspemu.hle.*
import com.soywiz.kpspemu.hle.manager.*
import com.soywiz.kpspemu.hle.modules.*
import com.soywiz.kpspemu.mem.*
import com.soywiz.krypto.encoding.*
import kotlin.coroutines.*
import kotlin.test.*

class IntegrationTests : BaseTest() {
    val TRACE = false
    val TRACE1 = false
    //val TRACE = true
    //val TRACE1 = true

    //val DEFAULT_MODE = Mode.Dynarek
    val DEFAULT_MODE = Mode.Interpreted
    val DEFAULT_TIMEOUT = 30.seconds


    //@Test fun testDmac() = testFile("dmac/dmactest")

    enum class Mode { Interpreted, Dynarek }

    @Test
    fun testCpuAlu() {
        //for (n in 0 until 1000) {
        for (n in 0 until 1) {
            testFile("cpu/cpu_alu/cpu_alu")
        }
    }

    //@Test
    //fun testCpuAluDynarek() = testFile("cpu/cpu_alu/cpu_alu", mode = Mode.Dynarek)

    @Test
    fun testCpuBranch() = testFile("cpu/cpu_alu/cpu_branch")

    @Test
    fun testCpuBranch2() = testFile("cpu/cpu_alu/cpu_branch2")

    @Test
    fun testIcache() = testFile("cpu/icache/icache")

    @Test
    fun testLsu() = testFile("cpu/lsu/lsu")

    @Test
    fun testFpu() {
        //for (n in 0 until 1000) {
        for (n in 0 until 1) {
            testFile(
                "cpu/fpu/fpu", ignores = listOf(
                    "mul.s 0.296558 * 62.000000, CAST_1 = 18.38657^",
                    "mul.s 0.296558 * 62.000000, FLOOR_3 = 18.38657^"
                )
            )
        }
    }

    @Test
    fun testFcr() = testFile(
        "cpu/fpu/fcr", ignores = listOf(
            "Underflow:\n  fcr0: 00003351, fcr25: 00000000, fcr26: 00000000, fcr27: 00000000, fcr28: 00000000, fcr31: ^^^^^^^^",
            "Inexact:\n  fcr0: 00003351, fcr25: 00000000, fcr26: 00000000, fcr27: 00000000, fcr28: 00000000, fcr31: ^^^^^^^^"
        )
    )

    //@Test fun testRtcRtc() = testFile("rtc/rtc")

    @Ignore // ("almost but fails in edge cases")
    @Test
    fun testRtcArithmetic() = testFile("rtc/arithmetic")

    @Test
    fun testThreadsK0() = testFile("threads/k0/k0")

    //@Test fun testKirk() = testFile("kirk/kirk")

    @Ignore
    @Test
    fun testLoaderBss() = testFile("loader/bss/bss")

    @Test
    fun testVfpuColors() = testFile("cpu/vfpu/colors")

    @Test
    fun testVfpuConvert() = testFile("cpu/vfpu/convert")

    @Test
    fun testVfpuVavg() = testFile("cpu/vfpu/vavg")

    @Ignore
    @Test
    fun testVfpuVector() = testFile("cpu/vfpu/vector")

    @Test
    fun testVfpuGum() = testFile("cpu/vfpu/gum")

    @Test
    fun testVfpuMatrix() = testFile("cpu/vfpu/matrix")

    @Test
    fun testVfpuPrefixes() = testFile("cpu/vfpu/prefixes")

    @Ignore
    @Test
    fun testVfpuVregs() = testFile("cpu/vfpu/vregs")

    @Ignore
    @Test
    fun testThreadsThreadsCreate() = testFile("threads/threads/create")

    @Ignore
    @Test
    fun testThreadsVplAllocate() = testFile("threads/vpl/allocate")

    @Test
    fun testThreadsVplVpl() = testFile("threads/vpl/vpl")

    fun testFile(
        name: String,
        ignores: List<String> = listOf(),
        mode: Mode = DEFAULT_MODE,
        timeout: TimeSpan = DEFAULT_TIMEOUT,
        processor: (String) -> String = { it }
    ) = pspSuspendTest(timeout) {
        testFile(
            elf = pspautotests["$name.prx"].readAsSyncStream(),
            expected = pspautotests["$name.expected"].readString(),
            ignores = ignores,
            mode = mode,
            timeout = timeout,
            processor = processor
        )
    }

    suspend fun testFile(
        elf: SyncStream,
        expected: String,
        ignores: List<String>,
        mode: Mode = DEFAULT_MODE,
        timeout: TimeSpan = DEFAULT_TIMEOUT,
        processor: (String) -> String = { it }
    ) {
        val emulator = Emulator(coroutineContext, mem = CachedMemory())

        emulator.interpreted = (mode == Mode.Interpreted)
        emulator.display.exposeDisplay = false
        emulator.registerNativeModules()
        //val info = emulator.loadElfAndSetRegisters(elf, "ms0:/PSP/GAME/EBOOT.PBP")
        emulator.fileManager.currentDirectory = "ms0:/PSP/GAME/virtual"
        emulator.fileManager.executableFile = "ms0:/PSP/GAME/virtual/EBOOT.PBP"
        emulator.deviceManager.mount(
            emulator.fileManager.currentDirectory,
            MemoryVfsMix("EBOOT.PBP" to elf.clone().toAsync())
        )
        val info = emulator.loadElfAndSetRegisters(elf, listOf("ms0:/PSP/GAME/virtual/EBOOT.PBP"))

        if (TRACE1) {
            Logger.defaultLevel = Logger.Level.TRACE
        }

        if (TRACE) {
            emulator.threadManager.trace("user_main", trace = true)
            Logger.defaultLevel = Logger.Level.TRACE
        } else {
            Logger.defaultLevel = Logger.Level.ERROR
            //Logger.setLevel("ElfPsp", Logger.Level.ERROR)
        }

        var generatedError: Throwable? = null

        //emulator.moduleManager.registerNativeModules().get<ThreadManForUser>().logger.setLevel(Logger.Level.TRACE)

        try {
            //println("[1]")
            withTimeout(timeout) {
                while (emulator.running) {
                    //println("[2] : ${emulator.running}")
                    val res = emulator.threadManager.step() // UPDATE THIS
                    delay(16.milliseconds)

                    if (res == ThreadManager.StepResult.NO_THREAD) {
                        println("res == ThreadManager.StepResult.NO_THREAD")
                        emulator.threadManager.dump()
                    }
                    //if (res == ThreadManager.StepResult.NO_THREAD) break
                    if (TRACE) {
                        for (thread in emulator.threadManager.threads) println("PC: ${thread.state.PC.hex} : ${(thread.state.PC - info.baseAddress).hex}")
                    }
                }
                println("emulator.running: ${emulator.running}")
            }
        } catch (e: Throwable) {
            Console.error("Partial output generated:")
            Console.error("'" + emulator.output.toString() + "'")
            //throw e
            e.printStackTrace()
            generatedError = e
        }

        val ignoresRegex = ignores.map {
            Regex(Regex.quote(it).replace("\\^", ".")) to it
        }

        fun String.normalize(): String {
            var out = this.replace("\r\n", "\n").replace("\r", "\n").trimEnd()
            for (rex in ignoresRegex) {
                out = out.replace(rex.first, rex.second)
            }
            return out
        }
        assertEquals(expected.normalize(), processor(emulator.output.toString().normalize()))
        //assertEquals(expected.normalize(), processor(emulator.output.toString().normalize()))
        generatedError?.let { throw it }
    }
}
