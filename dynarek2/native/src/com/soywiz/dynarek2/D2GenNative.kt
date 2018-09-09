package com.soywiz.dynarek2

import kotlinx.cinterop.*
import platform.posix.*

actual fun D2Func.generate(): D2Result {
    val bytes = com.soywiz.dynarek2.target.x64.Dynarek2X64Gen().generate(this)
    val funcMem = NewD2Memory(bytes)
    val cfunc = funcMem.mem.buffer.reinterpret<CFunction<(CPointer<ByteVar>?, CPointer<ByteVar>?, CPointer<ByteVar>?, Any?) -> Int>>()
    //val cfunc = mem.mem.buffer.reinterpret<CFunction<(CPointer<ByteVar>?, CPointer<ByteVar>?, CPointer<ByteVar>?, CPointer<ByteVar>?) -> Int>>()
    return D2Result(
        bytes,
        { regs: D2Memory?, mem: D2Memory?, temps: D2Memory?, external: Any? ->
            cfunc(regs?.ptr, mem?.ptr, temps?.ptr, external)
        }
    ) {
        funcMem.free()
    }
}
