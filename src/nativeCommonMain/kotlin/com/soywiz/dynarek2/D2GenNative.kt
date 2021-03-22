package com.soywiz.dynarek2

import kotlinx.cinterop.*
import platform.posix.*
import com.soywiz.dynarek2.target.x64.*
import kotlin.reflect.*

inline fun <reified R> D2Context.registerFunc(name: KFunction0<R>, address: Long) = _registerFunc(name, address, R::class.d2type)
inline fun <reified P1, reified R> D2Context.registerFunc(name: KFunction1<P1, R>, address: Long) = _registerFunc(name, address, R::class.d2type, P1::class.d2type)
inline fun <reified P1, reified P2, reified R> D2Context.registerFunc(name: KFunction2<P1, P2, R>, address: Long) = _registerFunc(name, address, R::class.d2type, P1::class.d2type, P2::class.d2type)
inline fun <reified P1, reified P2, reified P3, reified R> D2Context.registerFunc(name: KFunction3<P1, P2, P3, R>, address: Long) = _registerFunc(name, address, R::class.d2type, P1::class.d2type, P2::class.d2type, P3::class.d2type)
inline fun <reified P1, reified P2, reified P3, reified P4, reified R> D2Context.registerFunc(name: KFunction4<P1, P2, P3, P4, R>, address: Long) = _registerFunc(name, address, R::class.d2type, P1::class.d2type, P2::class.d2type, P3::class.d2type, P4::class.d2type)
inline fun <reified P1, reified P2, reified P3, reified P4, reified P5, reified R> D2Context.registerFunc(name: KFunction5<P1, P2, P3, P4, P5, R>, address: Long) = _registerFunc(name, address, R::class.d2type, P1::class.d2type, P2::class.d2type, P3::class.d2type, P4::class.d2type, P5::class.d2type)
inline fun <reified P1, reified P2, reified P3, reified P4, reified P5, reified P6, reified R> D2Context.registerFunc(name: KFunction6<P1, P2, P3, P4, P5, P6, R>, address: Long) = _registerFunc(name, address, R::class.d2type, P1::class.d2type, P2::class.d2type, P3::class.d2type, P4::class.d2type, P5::class.d2type, P6::class.d2type)

actual class D2Runner actual constructor() : D2BaseRunner() {
    var cfuncPtrLong = 0L
    var regsLong = 0L
    var memLong = 0L
    var tempsLong = 0L
    var externalLong = 0L
    var externalRef: StableRef<*>? = null

    override fun setParams(regs: D2Memory?, mem: D2Memory?, temps: D2Memory?, external: Any?) {
        close()
        regsLong = if (regs != null) regs.ptr.uncheckedCast<Long>() else 0L
        memLong = if (mem != null) mem.ptr.uncheckedCast<Long>() else 0L
        tempsLong = if (temps != null) temps.ptr.uncheckedCast<Long>() else 0L
        externalRef = if (external != null) StableRef.create(external) else null
        val externalPtr = externalRef?.asCPointer()
        externalLong = if (externalPtr != null) externalPtr.uncheckedCast<Long>() else 0L
    }

    override fun setFunc(result: D2Result) {
        cfuncPtrLong = result.extraLong
    }

    override fun execute(): Int {
        if (cfuncPtrLong == 0L) return 0
        return fastinvoke.invokeDynarekFast(cfuncPtrLong, regsLong, memLong, tempsLong, externalLong)
    }

    override fun close() {
        externalRef?.dispose()
        externalRef = null
    }
}

actual fun D2Func.generate(context: D2Context, name: String?, debug: Boolean): D2Result {
    val gen = Dynarek2X64Gen(context, name, debug)
    //val funcBytes = if (isNativeWindows) {
    //    gen.generateDummy()
    //} else {
    //    gen.generate(this)
    //}
    val funcBytes = gen.generate(this)
    val funcMem = NewD2Memory(funcBytes)
    val cfuncPtrLong = funcMem.mem.buffer.uncheckedCast<Long>()
    val cfunc = funcMem.mem.buffer.reinterpret<CFunction<(CPointer<ByteVar>?, CPointer<ByteVar>?, CPointer<ByteVar>?, COpaquePointer?) -> Int>>()

    if (name != null && debug) {
        fileWriteBytes("$name.bin", funcBytes)
    }

    return D2Result(
        name, debug,
        funcBytes,
        ::dummy_d2func,
        extraLong = cfuncPtrLong,
        free = {
            funcMem.free()
        }
    )
}

private fun dummy_d2func(regs: D2Memory?, mem: D2Memory?, temps: D2Memory?, external: Any?): Int {
    TODO("Deprecated: Use D2Runner instead")
}

//external fun invokeDynarekFast(address: Long, regs: Long, mem: Long, temps: Long, external: Long): Int

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

fun CPointer<ByteVar>.readBytes(count: Int): ByteArray {
    val out = ByteArray(count)
    for (n in 0 until count) out[n] = this[n]
    return out
}

fun _dyna_getDemo(ptr: COpaquePointer): Int {
    return dyna_getDemo(ptr.asStableRef<MyClass>().get())
}

val CPointer<*>.asLong get() = this.uncheckedCast<Long>()

actual fun D2Context.registerDefaultFunctions() {
    registerFunc(::iprint, staticCFunction(::iprint).asLong)
    registerFunc(::isqrt, staticCFunction(::isqrt).asLong)
    registerFunc(::isub, staticCFunction(::isub).asLong)
    registerFunc(::fsub, staticCFunction(::fsub).asLong)
    registerFunc(::demo_ithrow, staticCFunction(::demo_ithrow).asLong)
    registerFunc(::dyna_getDemo, staticCFunction(::_dyna_getDemo).asLong)

    //val isubptr = staticCFunction(::isub).reinterpret<ByteVar>()
    //for (n in 0 until 32) {
    //    println((isubptr[n].toInt() and 0xFF).toString(16))
    //}
    //registerFunc(Dynarek2X64Gen.SHL_NAME, staticCFunction(::_jit_shl).uncheckedCast<Long>())
    //registerFunc(Dynarek2X64Gen.SHR_NAME, staticCFunction(::_jit_shr).uncheckedCast<Long>())
    //registerFunc(Dynarek2X64Gen.USHR_NAME, staticCFunction(::_jit_ushr).uncheckedCast<Long>())
}

//fun _jit_shl(a: Int, b: Int): Int = a shl b
//fun _jit_shr(a: Int, b: Int): Int = a shr b
//fun _jit_ushr(a: Int, b: Int): Int = a ushr b
