package com.soywiz.kpspemu.debug

interface BaseBreakpoints {
    operator fun get(addr: Int): Boolean
}