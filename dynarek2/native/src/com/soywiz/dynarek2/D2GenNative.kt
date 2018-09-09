package com.soywiz.dynarek2

import kotlinx.cinterop.*
import platform.posix.*

actual fun D2Func.generate(name: String?, debug: Boolean): D2Result {
    val funcBytes = com.soywiz.dynarek2.target.x64.Dynarek2X64Gen().generate(this)
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
