package com.soywiz.dynarek2

import kotlinx.cinterop.*
import platform.posix.*
import com.soywiz.dynarek2.target.x64.*

actual fun D2Func.generate(context: D2Context, name: String?, debug: Boolean): D2Result {
    val funcBytes = Dynarek2X64Gen(context, name, debug).generate(this)
    val funcMem = NewD2Memory(funcBytes)
    val cfunc = funcMem.mem.buffer.reinterpret<CFunction<(CPointer<ByteVar>?, CPointer<ByteVar>?, CPointer<ByteVar>?, Any?) -> Int>>()
    //val cfunc = mem.mem.buffer.reinterpret<CFunction<(CPointer<ByteVar>?, CPointer<ByteVar>?, CPointer<ByteVar>?, CPointer<ByteVar>?) -> Int>>()

    if (name != null && debug) {
        fileWriteBytes("$name.bin", funcBytes)
    }

    return D2Result(
        funcBytes,
        { regs: D2Memory?, mem: D2Memory?, temps: D2Memory?, external: Any? ->
            cfunc(regs?.ptr, mem?.ptr, temps?.ptr, external)
        }
    ) {
        funcMem.free()
    }
}

fun fileWriteBytes(name: String, data: ByteArray) {
    val file = fopen(name, "wb")
    try {
        if (data.size > 0) {
            data.usePinned { pin ->
                fwrite(pin.addressOf(0), 1, data.size.convert(), file)
            }
        }
    } finally {
        fclose(file)
    }
}

actual fun D2Context.registerDefaultFunctions() {
    //registerFunc(Dynarek2X64Gen.SHL_NAME, staticCFunction(::_jit_shl).uncheckedCast<Long>())
    //registerFunc(Dynarek2X64Gen.SHR_NAME, staticCFunction(::_jit_shr).uncheckedCast<Long>())
    //registerFunc(Dynarek2X64Gen.USHR_NAME, staticCFunction(::_jit_ushr).uncheckedCast<Long>())
}

//fun _jit_shl(a: Int, b: Int): Int = a shl b
//fun _jit_shr(a: Int, b: Int): Int = a shr b
//fun _jit_ushr(a: Int, b: Int): Int = a ushr b
