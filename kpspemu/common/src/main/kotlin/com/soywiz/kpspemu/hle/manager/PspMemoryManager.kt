package com.soywiz.kpspemu.hle.manager

class PspMemoryManager(val space: IntRange) {
	class Entry {
		var pos: IntRange = 0..0
		var next: Entry? = null
	}
}