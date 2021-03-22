package com.soywiz.kpspemu.hle.modules

import com.soywiz.klock.*
import com.soywiz.korma.random.*
import com.soywiz.kpspemu.*
import com.soywiz.kpspemu.cpu.*
import com.soywiz.kpspemu.hle.*
import com.soywiz.kpspemu.mem.*
import com.soywiz.kpspemu.util.*
import kotlin.random.*

open class UtilsBase(emulator: Emulator, name: String, flags: Int, prx: String, prxName: String) :
    SceModule(emulator, name, flags, prx, prxName) {
    val random = Random(0)

    fun sceKernelUtilsMt19937Init(ctx: Ptr, seed: Int): Int {
        println("Not implemented UtilsForUser.sceKernelUtilsMt19937Init")
        return 0
    }

    fun sceKernelUtilsMt19937UInt(ctx: Ptr): Int {
        //println("Not implemented UtilsForUser.sceKernelUtilsMt19937UInt")
        val value = random.nextInt()
        //println("Random: $value")
        return value
    }

    // Time
    fun sceKernelLibcGettimeofday(timevalPtr: Ptr, timezonePtr: Ptr): Int {
        if (timevalPtr.isNotNull) {
            val totalMicroseconds = Klock.currentTimeMicroDouble()
            val totalSeconds = totalMicroseconds / 1_000_000
            val microseconds = totalMicroseconds % 1_000_000
            timevalPtr.sw(0, totalSeconds.toInt())
            timevalPtr.sw(4, microseconds.toInt())
        }
        if (timezonePtr.isNotNull) {
            val minutesWest = 0
            val dstTime = 0
            timevalPtr.sw(0, minutesWest)
            timevalPtr.sw(4, dstTime)
        }

        return 0
    }

    fun sceKernelLibcTime(ptr: Ptr): Int {
        val seconds = rtc.getTimeInSeconds()
        if (ptr.isNotNull) ptr.sw(0, seconds)
        return seconds
    }

    fun sceKernelLibcClock(): Int = timeManager.getTimeInMicrosecondsInt()


    // Data cache
    fun sceKernelDcacheWritebackAll(): Unit = emulator.dataCache(writeback = true, invalidate = false)

    fun sceKernelDcacheWritebackRange(ptr: Int, size: Int): Unit =
        emulator.dataCache(writeback = true, invalidate = false)

    fun sceKernelDcacheWritebackInvalidateAll(): Unit = emulator.dataCache(writeback = true, invalidate = true)
    fun sceKernelDcacheWritebackInvalidateRange(ptr: Int, size: Int): Unit =
        emulator.dataCache(ptr, size, writeback = true, invalidate = true)

    fun sceKernelDcacheInvalidateRange(ptr: Int, size: Int): Unit =
        emulator.dataCache(ptr, size, writeback = false, invalidate = true)

    // Instruction cache
    fun sceKernelIcacheInvalidateRange(ptr: Int, size: Int): Unit = emulator.invalidateInstructionCache(ptr, size)

    fun sceKernelIcacheInvalidateAll(): Unit = emulator.invalidateInstructionCache()

    // Cache tags
    fun sceKernelDcacheReadTag(cpu: CpuState): Unit = UNIMPLEMENTED(0x16641D70)

    fun sceKernelIcacheProbe(cpu: CpuState): Unit = UNIMPLEMENTED(0x4FD31C9D)
    fun sceKernelDcacheProbeRange(cpu: CpuState): Unit = UNIMPLEMENTED(0x77DFF087)
    fun sceKernelDcacheProbe(cpu: CpuState): Unit = UNIMPLEMENTED(0x80001C4C)
    fun sceKernelIcacheReadTag(cpu: CpuState): Unit = UNIMPLEMENTED(0xFB05FAD0)

    fun UtilsForUser_004D4DEE(cpu: CpuState): Unit = UNIMPLEMENTED(0x004D4DEE)
    fun sceKernelUtilsMt19937UInt(cpu: CpuState): Unit = UNIMPLEMENTED(0x06FB8A63)
    fun UtilsForUser_157A383A(cpu: CpuState): Unit = UNIMPLEMENTED(0x157A383A)
    fun UtilsForUser_1B0592A3(cpu: CpuState): Unit = UNIMPLEMENTED(0x1B0592A3)
    fun sceKernelUtilsSha1BlockUpdate(cpu: CpuState): Unit = UNIMPLEMENTED(0x346F6DA8)
    fun sceKernelGetGPI(cpu: CpuState): Unit = UNIMPLEMENTED(0x37FB5C42)
    fun UtilsForUser_39F49610(cpu: CpuState): Unit = UNIMPLEMENTED(0x39F49610)
    fun UtilsForUser_3FD3D324(cpu: CpuState): Unit = UNIMPLEMENTED(0x3FD3D324)
    fun UtilsForUser_43C9A8DB(cpu: CpuState): Unit = UNIMPLEMENTED(0x43C9A8DB)
    fun UtilsForUser_515B4FAF(cpu: CpuState): Unit = UNIMPLEMENTED(0x515B4FAF)
    fun sceKernelUtilsSha1BlockResult(cpu: CpuState): Unit = UNIMPLEMENTED(0x585F1C09)
    fun UtilsForUser_5C7F2B1A(cpu: CpuState): Unit = UNIMPLEMENTED(0x5C7F2B1A)
    fun sceKernelUtilsMd5BlockUpdate(cpu: CpuState): Unit = UNIMPLEMENTED(0x61E1E525)
    fun UtilsForUser_6231A71D(cpu: CpuState): Unit = UNIMPLEMENTED(0x6231A71D)
    fun sceKernelSetGPO(cpu: CpuState): Unit = UNIMPLEMENTED(0x6AD345D7)
    fun UtilsForUser_7333E539(cpu: CpuState): Unit = UNIMPLEMENTED(0x7333E539)
    fun UtilsForUser_740DF7F0(cpu: CpuState): Unit = UNIMPLEMENTED(0x740DF7F0)
    fun sceKernelUtilsSha1Digest(cpu: CpuState): Unit = UNIMPLEMENTED(0x840259F1)
    fun sceKernelPutUserLog(cpu: CpuState): Unit = UNIMPLEMENTED(0x87E81561)
    fun UtilsForUser_99134C3F(cpu: CpuState): Unit = UNIMPLEMENTED(0x99134C3F)
    fun sceKernelUtilsMd5BlockInit(cpu: CpuState): Unit = UNIMPLEMENTED(0x9E5C5086)
    fun UtilsForUser_AF3766BB(cpu: CpuState): Unit = UNIMPLEMENTED(0xAF3766BB)
    fun UtilsForUser_B83A1E76(cpu: CpuState): Unit = UNIMPLEMENTED(0xB83A1E76)
    fun sceKernelUtilsMd5BlockResult(cpu: CpuState): Unit = UNIMPLEMENTED(0xB8D24E78)
    fun sceKernelUtilsMd5Digest(cpu: CpuState): Unit = UNIMPLEMENTED(0xC8186A58)
    fun UtilsForUser_DBBE9A46(cpu: CpuState): Unit = UNIMPLEMENTED(0xDBBE9A46)
    fun sceKernelUtilsMt19937Init(cpu: CpuState): Unit = UNIMPLEMENTED(0xE860E75E)
    fun UtilsForUser_F0155BCA(cpu: CpuState): Unit = UNIMPLEMENTED(0xF0155BCA)
    fun sceKernelUtilsSha1BlockInit(cpu: CpuState): Unit = UNIMPLEMENTED(0xF8FCD5BA)


    override fun registerModule() {
        // Rand
        registerFunctionInt(
            "sceKernelUtilsMt19937Init",
            0xE860E75E,
            since = 150,
            syscall = 0x20BF
        ) { sceKernelUtilsMt19937Init(ptr, int) }
        registerFunctionInt(
            "sceKernelUtilsMt19937UInt",
            0x06FB8A63,
            since = 150,
            syscall = 0x20C0
        ) { sceKernelUtilsMt19937UInt(ptr) }

        // Time
        registerFunctionInt("sceKernelLibcGettimeofday", 0x71EC4271, since = 150) {
            sceKernelLibcGettimeofday(
                ptr,
                ptr
            )
        }
        registerFunctionInt("sceKernelLibcTime", 0x27CC57F0, since = 150) { sceKernelLibcTime(ptr) }
        registerFunctionInt("sceKernelLibcClock", 0x91E4F6A7, since = 150) { sceKernelLibcClock() }

        // Cache
        registerFunctionVoid("sceKernelDcacheWritebackAll", 0x79D1C3FA, since = 150) { sceKernelDcacheWritebackAll() }
        registerFunctionVoid("sceKernelIcacheInvalidateAll", 0x920F104A, since = 150) { sceKernelIcacheInvalidateAll() }
        registerFunctionVoid(
            "sceKernelDcacheWritebackInvalidateAll",
            0xB435DEC5,
            since = 150
        ) { sceKernelDcacheWritebackInvalidateAll() }
        registerFunctionVoid(
            "sceKernelDcacheInvalidateRange",
            0xBFA98062,
            since = 150
        ) { sceKernelDcacheInvalidateRange(int, int) }
        registerFunctionVoid("sceKernelDcacheWritebackRange", 0x3EE30821, since = 150) {
            sceKernelDcacheWritebackRange(
                int,
                int
            )
        }
        registerFunctionVoid(
            "sceKernelDcacheWritebackInvalidateRange",
            0x34B9FA9E,
            since = 150
        ) { sceKernelDcacheWritebackInvalidateRange(int, int) }
        registerFunctionVoid(
            "sceKernelIcacheInvalidateRange",
            0xC2DF770E,
            since = 150
        ) { sceKernelIcacheInvalidateRange(int, int) }

        registerFunctionRaw("UtilsForUser_004D4DEE", 0x004D4DEE, since = 150) { UtilsForUser_004D4DEE(it) }
        registerFunctionRaw("UtilsForUser_157A383A", 0x157A383A, since = 150) { UtilsForUser_157A383A(it) }
        registerFunctionRaw("sceKernelDcacheReadTag", 0x16641D70, since = 150) { sceKernelDcacheReadTag(it) }
        registerFunctionRaw("UtilsForUser_1B0592A3", 0x1B0592A3, since = 150) { UtilsForUser_1B0592A3(it) }
        registerFunctionRaw(
            "sceKernelUtilsSha1BlockUpdate",
            0x346F6DA8,
            since = 150
        ) { sceKernelUtilsSha1BlockUpdate(it) }
        registerFunctionRaw("sceKernelGetGPI", 0x37FB5C42, since = 150) { sceKernelGetGPI(it) }
        registerFunctionRaw("UtilsForUser_39F49610", 0x39F49610, since = 150) { UtilsForUser_39F49610(it) }
        registerFunctionRaw("UtilsForUser_3FD3D324", 0x3FD3D324, since = 150) { UtilsForUser_3FD3D324(it) }
        registerFunctionRaw("UtilsForUser_43C9A8DB", 0x43C9A8DB, since = 150) { UtilsForUser_43C9A8DB(it) }
        registerFunctionRaw("sceKernelIcacheProbe", 0x4FD31C9D, since = 150) { sceKernelIcacheProbe(it) }
        registerFunctionRaw("UtilsForUser_515B4FAF", 0x515B4FAF, since = 150) { UtilsForUser_515B4FAF(it) }
        registerFunctionRaw(
            "sceKernelUtilsSha1BlockResult",
            0x585F1C09,
            since = 150
        ) { sceKernelUtilsSha1BlockResult(it) }
        registerFunctionRaw("UtilsForUser_5C7F2B1A", 0x5C7F2B1A, since = 150) { UtilsForUser_5C7F2B1A(it) }
        registerFunctionRaw(
            "sceKernelUtilsMd5BlockUpdate",
            0x61E1E525,
            since = 150
        ) { sceKernelUtilsMd5BlockUpdate(it) }
        registerFunctionRaw("UtilsForUser_6231A71D", 0x6231A71D, since = 150) { UtilsForUser_6231A71D(it) }
        registerFunctionRaw("sceKernelSetGPO", 0x6AD345D7, since = 150) { sceKernelSetGPO(it) }
        registerFunctionRaw("UtilsForUser_7333E539", 0x7333E539, since = 150) { UtilsForUser_7333E539(it) }
        registerFunctionRaw("UtilsForUser_740DF7F0", 0x740DF7F0, since = 150) { UtilsForUser_740DF7F0(it) }
        registerFunctionRaw("sceKernelDcacheProbeRange", 0x77DFF087, since = 150) { sceKernelDcacheProbeRange(it) }
        registerFunctionRaw("sceKernelDcacheProbe", 0x80001C4C, since = 150) { sceKernelDcacheProbe(it) }
        registerFunctionRaw("sceKernelUtilsSha1Digest", 0x840259F1, since = 150) { sceKernelUtilsSha1Digest(it) }
        registerFunctionRaw("sceKernelPutUserLog", 0x87E81561, since = 150) { sceKernelPutUserLog(it) }
        registerFunctionRaw("UtilsForUser_99134C3F", 0x99134C3F, since = 150) { UtilsForUser_99134C3F(it) }
        registerFunctionRaw("sceKernelUtilsMd5BlockInit", 0x9E5C5086, since = 150) { sceKernelUtilsMd5BlockInit(it) }
        registerFunctionRaw("UtilsForUser_AF3766BB", 0xAF3766BB, since = 150) { UtilsForUser_AF3766BB(it) }
        registerFunctionRaw("UtilsForUser_B83A1E76", 0xB83A1E76, since = 150) { UtilsForUser_B83A1E76(it) }
        registerFunctionRaw(
            "sceKernelUtilsMd5BlockResult",
            0xB8D24E78,
            since = 150
        ) { sceKernelUtilsMd5BlockResult(it) }
        registerFunctionRaw("sceKernelUtilsMd5Digest", 0xC8186A58, since = 150) { sceKernelUtilsMd5Digest(it) }
        registerFunctionRaw("UtilsForUser_DBBE9A46", 0xDBBE9A46, since = 150) { UtilsForUser_DBBE9A46(it) }
        registerFunctionRaw("UtilsForUser_F0155BCA", 0xF0155BCA, since = 150) { UtilsForUser_F0155BCA(it) }
        registerFunctionRaw("sceKernelUtilsSha1BlockInit", 0xF8FCD5BA, since = 150) { sceKernelUtilsSha1BlockInit(it) }
        registerFunctionRaw("sceKernelIcacheReadTag", 0xFB05FAD0, since = 150) { sceKernelIcacheReadTag(it) }
    }
}