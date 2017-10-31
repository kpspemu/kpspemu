package com.soywiz.kpspemu.ctrl

import com.soywiz.kpspemu.Emulator
import com.soywiz.kpspemu.WithEmulator
import com.soywiz.kpspemu.util.clamp

class PspController(override val emulator: Emulator) : WithEmulator {
	var buttons: Int = 0
	var lx: Int = 128
	var ly: Int = 128

	fun updateButton(button: PspCtrlButtons, pressed: Boolean) {
		if (pressed) {
			this.buttons = this.buttons or button.bits
		} else {
			this.buttons = this.buttons and button.bits.inv()
		}
	}

	private fun fixFloat(v: Float): Int = ((v.clamp(-1f, 1f) * 127) + 128).toInt()

	fun updateAnalog(x: Float, y: Float) {
		this.lx = fixFloat(x)
		this.ly = fixFloat(y)
		//println("Update analog: ($x, $y) - ($lx, $ly)")
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