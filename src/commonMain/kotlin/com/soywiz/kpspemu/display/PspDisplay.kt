package com.soywiz.kpspemu.display

import com.soywiz.kmem.*
import com.soywiz.korim.bitmap.*
import com.soywiz.korim.color.*
import com.soywiz.kpspemu.*
import com.soywiz.kpspemu.ge.*
import com.soywiz.kpspemu.hle.manager.*
import com.soywiz.kpspemu.mem.*

class PspDisplay(override val emulator: Emulator) : WithEmulator {
    companion object {
        const val PROCESSED_PIXELS_PER_SECOND = 9000000 // hz
        const val CYCLES_PER_PIXEL = 1.0
        const val PIXELS_IN_A_ROW = 525
        const val VSYNC_ROW = 272
        const val NUMBER_OF_ROWS = 286
        const val HCOUNT_PER_VBLANK = 285.72
        const val HORIZONTAL_SYNC_HZ =
            (PspDisplay.PROCESSED_PIXELS_PER_SECOND.toDouble() * PspDisplay.CYCLES_PER_PIXEL.toDouble()) / PspDisplay.PIXELS_IN_A_ROW.toDouble()// 17142.85714285714
        const val HORIZONTAL_SECONDS = 1.0 / PspDisplay.HORIZONTAL_SYNC_HZ // 5.8333333333333E-5
        const val VERTICAL_SYNC_HZ = PspDisplay.HORIZONTAL_SYNC_HZ / PspDisplay.HCOUNT_PER_VBLANK // 59.998800024
        const val VERTICAL_SECONDS = 1.0 / PspDisplay.VERTICAL_SYNC_HZ // 0.016667
        const val VERTICAL_MS = VERTICAL_SECONDS * 1000.0 // 16.667
    }

    val bmp = Bitmap32(480, 272)
    var exposeDisplay = true
    var rawDisplay: Boolean = true
    var address: Int = 0x44000000
    var bufferWidth: Int = 512
    var pixelFormat: PixelFormat = PixelFormat.RGBA_8888
    var sync: Int = 0
    var displayMode: Int = 0
    var displayWidth: Int = 512
    var displayHeight: Int = 272
    private val temp = ByteArray(512 * 272 * 4)

    fun fixedAddress(): Int {
        //println(address.hex)
        return address
    }

    fun decodeToBitmap32(out: Bitmap32) {
        val bmpData = out.data

        when (pixelFormat) {
            PixelFormat.RGBA_8888 -> { // Optimized!
                for (n in 0 until 272) {
                    mem.read(address + n * 512 * 4, bmpData.ints, bmp.width * n, bmp.width)
                }
            }
            else -> {
                mem.read(address, temp, 0, temp.size)
                val color = when (pixelFormat) {
                //PixelFormat.RGBA_5650 -> RGBA
                    PixelFormat.RGBA_5551 -> RGBA_5551
                    else -> RGBA
                }
                val stride = 512 * color.bytesPerPixel

                for (n in 0 until 272) {
                    color.decode(temp, (n * stride).toInt(), bmpData, bmp.width * n, bmp.width)
                }
            }
        }
    }

    fun crash() = invertDisplay()

    fun invertDisplay() {
        for (n in 0 until 512 * 272) {
            mem.sw(address + n * 4, mem.lw(address + n * 4).inv())
        }
        //mem.fill(0, Memory.VIDEOMEM.start, Memory.VIDEOMEM.size)
    }

    val times = DisplayTimes { emulator.timeManager.getTimeInMillisecondsDouble() }

    val updatedTimes get() = run { times.updateTime(); times }

    suspend fun waitVblankStart(thread: PspThread, reason: String) {
        times.updateTime()
        thread.sleepSecondsIfRequired(times.secondsLeftForVblankStart)
    }

    suspend fun waitVblank(thread: PspThread, reason: String) {
        times.updateTime()
        thread.sleepSecondsIfRequired(times.secondsLeftForVblank)
    }

    fun reset() {
        times.reset()
        exposeDisplay = true
        rawDisplay = true
        address = 0x44000000
        bufferWidth = 512
        pixelFormat = PixelFormat.RGBA_8888
        sync = 0
        displayMode = 0
        displayWidth = 480
        displayHeight = 272
        bmp.fill(Colors.TRANSPARENT_BLACK)
        temp.fill(0)
    }
}

class DisplayTimes(val msProvider: () -> Double) {
    private var startTimeMs: Double = 0.0
    private var currentMs: Double = startTimeMs
    private var elapsedSeconds: Double = 0.0
    var hcountTotal = 0
    var hcountCurrent = 0
    var vblankCount = 0
    private var isInVblank = false

    private var rowsLeftForVblank = 0
    var secondsLeftForVblank = 0.0

    private var rowsLeftForVblankStart = 0
    var secondsLeftForVblankStart = 0.0

    fun updateTime(now: Double = msProvider()) {
        if (startTimeMs == 0.0) startTimeMs = now
        this.currentMs = now
        this.elapsedSeconds = (this.currentMs - this.startTimeMs) / 1000
        this.hcountTotal = (this.elapsedSeconds * PspDisplay.HORIZONTAL_SYNC_HZ).toInt()
        this.hcountCurrent = (((this.elapsedSeconds % 1.00002) * PspDisplay.HORIZONTAL_SYNC_HZ).toInt()) %
                PspDisplay.NUMBER_OF_ROWS
        this.vblankCount = (this.elapsedSeconds * PspDisplay.VERTICAL_SYNC_HZ).toInt()
        //console.log(this.elapsedSeconds);
        if (this.hcountCurrent >= PspDisplay.VSYNC_ROW) {
            this.isInVblank = true
            this.rowsLeftForVblank = 0
            this.rowsLeftForVblankStart = (PspDisplay.NUMBER_OF_ROWS - this.hcountCurrent) + PspDisplay.VSYNC_ROW
        } else {
            this.isInVblank = false
            this.rowsLeftForVblank = PspDisplay.VSYNC_ROW - this.hcountCurrent
            this.rowsLeftForVblankStart = this.rowsLeftForVblank
        }
        this.secondsLeftForVblank = this.rowsLeftForVblank * PspDisplay.HORIZONTAL_SECONDS
        this.secondsLeftForVblankStart = this.rowsLeftForVblankStart * PspDisplay.HORIZONTAL_SECONDS
    }

    fun reset() {
        // TODO?
        startTimeMs = 0.0
    }

}