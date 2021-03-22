package com.soywiz.kpspemu.ctrl

import com.soywiz.kmem.*
import com.soywiz.kpspemu.*
import com.soywiz.kpspemu.util.*

class PspController(override val emulator: Emulator) : WithEmulator {
    var samplingCycle: Int = 0
    var samplingMode: Int = 0
    val frames = (0 until 0x10).map { Frame() }
    var frameIndex = 0
    fun getFrame(offset: Int): Frame = frames[(frameIndex + offset) umod frames.size]
    val lastLatchData = Frame()

    data class Frame(var timestamp: Int = 0, var buttons: Int = 0, var lx: Int = 128, var ly: Int = 128) {
        fun setTo(other: Frame) {
            this.buttons = other.buttons
            this.lx = other.lx
            this.ly = other.ly
        }

        //fun press(button: PspCtrlButtons) {
        //    buttons = buttons or button.bits
        //}
//
        //fun release(button: PspCtrlButtons) {
        //    buttons = buttons and button.bits.inv()
        //}
//
        //fun update(button: PspCtrlButtons, pressed: Boolean) {
        //    if (pressed) press(button) else release(button)
        //}

        fun reset() {
            timestamp = 0
            buttons = 0
            lx = 128
            ly = 128
        }
    }

    val currentFrame get() = frames[frameIndex]

    fun startFrame(timestamp: Int) = run { currentFrame.timestamp = timestamp }

    fun endFrame() {
        val lastFrame = currentFrame
        frameIndex = (frameIndex + 1) % frames.size
        currentFrame.setTo(lastFrame)
    }

    fun updateButton(button: PspCtrlButtons, pressed: Boolean) {
        currentFrame.buttons = currentFrame.buttons.setBits(button.bits, pressed)
        //if (button == PspCtrlButtons.home && pressed) {
        //    emulator.onHomePress(Unit)
        //}
        //if (button == PspCtrlButtons.hold && pressed) {
        //    emulator.onLoadPress(Unit)
        //}
    }

    private fun fixFloat(v: Float): Int = ((v.clamp(-1f, 1f) * 127) + 128).toInt()

    fun updateAnalog(x: Float, y: Float) {
        currentFrame.lx = fixFloat(x)
        currentFrame.ly = fixFloat(y)
        //println("Update analog: ($x, $y) - ($lx, $ly)")
    }

    fun reset() {
        samplingCycle = 0
        samplingMode = 0
        for (f in frames) f.reset()
        lastLatchData.reset()
        frameIndex = 0
    }
}

enum class PspCtrlButtons(val bits: Int) { //: uint
    none(0x0000000),
    select(0x0000001),
    start(0x0000008),
    up(0x0000010),
    right(0x0000020),
    down(0x0000040),
    left(0x0000080),
    leftTrigger(0x0000100),
    rightTrigger(0x0000200),
    triangle(0x0001000),
    circle(0x0002000),
    cross(0x0004000),
    square(0x0008000),
    home(0x0010000),
    hold(0x0020000),
    wirelessLanUp(0x0040000),
    remote(0x0080000),
    volumeUp(0x0100000),
    volumeDown(0x0200000),
    screen(0x0400000),
    note(0x0800000),
    discPresent(0x1000000),
    memoryStickPresent(0x2000000);
}