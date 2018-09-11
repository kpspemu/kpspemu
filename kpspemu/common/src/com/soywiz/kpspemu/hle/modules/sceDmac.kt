package com.soywiz.kpspemu.hle.modules

import com.soywiz.kpspemu.*
import com.soywiz.kpspemu.hle.*
import com.soywiz.kpspemu.hle.error.*
import com.soywiz.kpspemu.mem.*

class sceDmac(emulator: Emulator) : SceModule(emulator, "sceDmac", 0x40010011, "lowio.prx", "sceLowIO_Driver") {
    fun _sceDmacMemcpy(src: Int, dst: Int, size: Int): Int {
        if (size == 0) return SceKernelErrors.ERROR_INVALID_SIZE
        if (src == 0 || dst == 0) return SceKernelErrors.ERROR_INVALID_POINTER
        mem.copy(dst, src, size)
        return 0
    }

    fun sceDmacMemcpy(src: Int, dst: Int, size: Int): Int = _sceDmacMemcpy(src, dst, size)

    fun sceDmacTryMemcpy(src: Int, dst: Int, size: Int): Int = _sceDmacMemcpy(src, dst, size)

    override fun registerModule() {
        registerFunctionInt("sceDmacMemcpy", 0x617F3FE6, since = 150) { sceDmacMemcpy(int, int, int) }
        registerFunctionInt("sceDmacTryMemcpy", 0xD97F94D8, since = 150) { sceDmacTryMemcpy(int, int, int) }
    }
}
