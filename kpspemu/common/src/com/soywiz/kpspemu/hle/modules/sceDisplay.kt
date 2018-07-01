package com.soywiz.kpspemu.hle.modules

import com.soywiz.kpspemu.*
import com.soywiz.kpspemu.cpu.*
import com.soywiz.kpspemu.ge.*
import com.soywiz.kpspemu.hle.*
import com.soywiz.kpspemu.hle.manager.*
import com.soywiz.kpspemu.mem.*

@Suppress("UNUSED_PARAMETER")
class sceDisplay(emulator: Emulator) :
    SceModule(emulator, "sceDisplay", 0x40010011, "display_02g.prx", "sceDisplay_Service") {

    fun sceDisplaySetMode(mode: Int, width: Int, height: Int): Int {
        //console.info(sprintf("sceDisplay.sceDisplaySetMode(mode: %d, width: %d, height: %d)", mode, width, height));
        display.displayMode = mode;
        display.displayWidth = width
        display.displayHeight = height
        return 0
    }

    suspend fun sceDisplayWaitVblankStart(thread: PspThread): Int {
        display.waitVblankStart(thread, "sceDisplayWaitVblankStart")
        return 0
    }

    suspend fun sceDisplayWaitVblank(thread: PspThread): Int {
        display.waitVblank(thread, "sceDisplayWaitVblank")
        return 0
    }

    fun sceDisplaySetFrameBuf(address: Int, bufferWidth: Int, pixelFormat: Int, sync: Int): Int {
        // PixelFormat
        //println("display.address: $address")
        display.address = address
        display.bufferWidth = bufferWidth
        display.pixelFormat = PixelFormat(pixelFormat)
        display.sync = sync
        return 0
    }

    fun sceDisplayGetFrameBuf(topaddrAddr: Ptr32, bufferwidthAddr: Ptr32, pixelformatAddr: Ptr32, syncType: Int): Int {
        topaddrAddr.set(display.address)
        bufferwidthAddr.set(display.bufferWidth)
        pixelformatAddr.set(display.pixelFormat.id)
        return 0
    }

    fun sceDisplayGetVcount(): Int = emulator.display.updatedTimes.vblankCount
    fun sceDisplayGetCurrentHcount(): Int = emulator.display.updatedTimes.hcountCurrent

    fun sceDisplayIsVsync(cpu: CpuState): Unit = UNIMPLEMENTED(0x21038913)
    fun sceDisplayGetAccumulatedHcount(cpu: CpuState): Unit = UNIMPLEMENTED(0x210EAB3A)
    fun sceDisplayGetBrightness(cpu: CpuState): Unit = UNIMPLEMENTED(0x31C4BAA8)
    fun sceDisplay_40F1469C(cpu: CpuState): Unit = UNIMPLEMENTED(0x40F1469C)
    fun sceDisplayIsVblank(cpu: CpuState): Unit = UNIMPLEMENTED(0x4D4E10EC)
    fun sceDisplayGetVblankRest(cpu: CpuState): Unit = UNIMPLEMENTED(0x69B53541)
    fun sceDisplay_77ED8B3A(cpu: CpuState): Unit = UNIMPLEMENTED(0x77ED8B3A)
    fun sceDisplaySetHoldMode(cpu: CpuState): Unit = UNIMPLEMENTED(0x7ED59BC4)
    fun sceDisplaySetResumeMode(cpu: CpuState): Unit = UNIMPLEMENTED(0xA544C486)
    fun sceDisplayAdjustAccumulatedHcount(cpu: CpuState): Unit = UNIMPLEMENTED(0xA83EF139)
    fun sceDisplayIsForeground(cpu: CpuState): Unit = UNIMPLEMENTED(0xB4F378FA)
    fun sceDisplayGetResumeMode(cpu: CpuState): Unit = UNIMPLEMENTED(0xBF79F646)
    fun sceDisplayGetFramePerSec(cpu: CpuState): Unit = UNIMPLEMENTED(0xDBA6C4C4)
    fun sceDisplayGetMode(cpu: CpuState): Unit = UNIMPLEMENTED(0xDEA197D4)


    override fun registerModule() {
        registerFunctionInt("sceDisplaySetMode", 0x0E20F177, 150, syscall = 0x213A) { sceDisplaySetMode(int, int, int) }
        registerFunctionInt("sceDisplaySetFrameBuf", 0x289D82FE, 150, syscall = 0x213F) {
            sceDisplaySetFrameBuf(
                int,
                int,
                int,
                int
            )
        }
        registerFunctionSuspendInt("sceDisplayWaitVblank", 0x36CDFADE, since = 150, cb = false) {
            sceDisplayWaitVblank(
                thread
            )
        }
        registerFunctionSuspendInt("sceDisplayWaitVblankCB", 0x8EB9EC49, since = 150, cb = true) {
            sceDisplayWaitVblank(
                thread
            )
        }
        registerFunctionSuspendInt(
            "sceDisplayWaitVblankStart",
            0x984C27E7,
            150,
            syscall = 0x2147,
            cb = false
        ) { sceDisplayWaitVblankStart(thread) }
        registerFunctionSuspendInt(
            "sceDisplayWaitVblankStartCB",
            0x46F186C3,
            since = 150,
            cb = true
        ) { sceDisplayWaitVblankStart(thread) }
        registerFunctionInt("sceDisplayGetVcount", 0x9C6EAAD7, since = 150) { sceDisplayGetVcount() }
        registerFunctionInt("sceDisplayGetCurrentHcount", 0x773DD3A3, since = 150) { sceDisplayGetCurrentHcount() }
        registerFunctionInt("sceDisplayGetFrameBuf", 0xEEDA2E54, since = 150) {
            sceDisplayGetFrameBuf(
                ptr32,
                ptr32,
                ptr32,
                int
            )
        }

        registerFunctionRaw("sceDisplayIsVsync", 0x21038913, since = 150) { sceDisplayIsVsync(it) }
        registerFunctionRaw("sceDisplayGetAccumulatedHcount", 0x210EAB3A, since = 150) {
            sceDisplayGetAccumulatedHcount(
                it
            )
        }
        registerFunctionRaw("sceDisplayGetBrightness", 0x31C4BAA8, since = 150) { sceDisplayGetBrightness(it) }
        registerFunctionRaw("sceDisplay_40F1469C", 0x40F1469C, since = 150) { sceDisplay_40F1469C(it) }
        registerFunctionRaw("sceDisplayIsVblank", 0x4D4E10EC, since = 150) { sceDisplayIsVblank(it) }
        registerFunctionRaw("sceDisplayGetVblankRest", 0x69B53541, since = 150) { sceDisplayGetVblankRest(it) }
        registerFunctionRaw("sceDisplay_77ED8B3A", 0x77ED8B3A, since = 150) { sceDisplay_77ED8B3A(it) }
        registerFunctionRaw("sceDisplaySetHoldMode", 0x7ED59BC4, since = 150) { sceDisplaySetHoldMode(it) }
        registerFunctionRaw("sceDisplaySetResumeMode", 0xA544C486, since = 150) { sceDisplaySetResumeMode(it) }
        registerFunctionRaw(
            "sceDisplayAdjustAccumulatedHcount",
            0xA83EF139,
            since = 150
        ) { sceDisplayAdjustAccumulatedHcount(it) }
        registerFunctionRaw("sceDisplayIsForeground", 0xB4F378FA, since = 150) { sceDisplayIsForeground(it) }
        registerFunctionRaw("sceDisplayGetResumeMode", 0xBF79F646, since = 150) { sceDisplayGetResumeMode(it) }
        registerFunctionRaw("sceDisplayGetFramePerSec", 0xDBA6C4C4, since = 150) { sceDisplayGetFramePerSec(it) }
        registerFunctionRaw("sceDisplayGetMode", 0xDEA197D4, since = 150) { sceDisplayGetMode(it) }
    }
}
