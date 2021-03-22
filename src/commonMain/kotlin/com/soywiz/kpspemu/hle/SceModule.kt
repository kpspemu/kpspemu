package com.soywiz.kpspemu.hle

import com.soywiz.kds.*
import com.soywiz.klogger.*
import com.soywiz.kmem.*
import com.soywiz.korio.error.*
import com.soywiz.korio.lang.*
import com.soywiz.kpspemu.*
import com.soywiz.kpspemu.cpu.*
import com.soywiz.kpspemu.hle.error.*
import com.soywiz.kpspemu.hle.manager.*
import com.soywiz.kpspemu.mem.*
import com.soywiz.kpspemu.util.*
import com.soywiz.krypto.encoding.*
import kotlin.coroutines.*
import com.soywiz.korio.lang.invalidOp as invalidOp1
import kotlin.coroutines.intrinsics.*

class RegisterReader {
    var pos: Int = 4
    lateinit var emulator: Emulator
    lateinit var cpu: CpuState

    fun reset(cpu: CpuState) {
        this.cpu = cpu
        this.pos = 4
    }

    val thread: PspThread get() = cpu.thread
    val mem: Memory get() = cpu.mem
    val bool: Boolean get() = int != 0
    val int: Int get() = this.cpu.getGpr(pos++)
    val long: Long
        get() {
            pos = pos.nextAlignedTo(2) // Ensure register alignment
            val low = this.cpu.getGpr(pos++)
            val high = this.cpu.getGpr(pos++)
            return (high.toLong() shl 32) or (low.toLong() and 0xFFFFFFFF)
        }
    val ptr: Ptr get() = MemPtr(mem, int)
    val ptr8: Ptr8 get() = Ptr8(ptr)
    val ptr32: Ptr32 get() = Ptr32(ptr)
    val ptr64: Ptr64 get() = Ptr64(ptr)
    val str: String? get() = mem.readStringzOrNull(int)
    val istr: String get() = mem.readStringzOrNull(int) ?: ""
    val strnn: String get() = mem.readStringzOrNull(int) ?: ""
    fun <T> ptr(s: StructType<T>) = PtrStruct(s, ptr)
}

data class NativeFunction(
    val name: String,
    val nid: Long,
    val since: Int,
    val syscall: Int,
    val function: (CpuState) -> Unit
)

abstract class BaseSceModule {
    abstract val mmodule: SceModule
    abstract val name: String

    fun getByNidOrNull(nid: Int): NativeFunction? = mmodule.functions[nid]
    fun getByNid(nid: Int): NativeFunction =
        getByNidOrNull(nid) ?: invalidOp1("Can't find NID 0x%08X in %s".format(nid, name))

    fun UNIMPLEMENTED(nid: Int): Nothing {
        val func = getByNid(nid)
        TODO("Unimplemented %s:0x%08X:%s".format(this.name, func.nid, func.name))
    }

    fun UNIMPLEMENTED(nid: Long): Nothing = UNIMPLEMENTED(nid.toInt())
}

abstract class SceSubmodule<out T : SceModule>(override val mmodule: T) : WithEmulator, BaseSceModule() {
    override val name: String get() = mmodule.name
    override val emulator: Emulator get() = mmodule.emulator
}

abstract class SceModule(
    override val emulator: Emulator,
    override val name: String,
    val flags: Int = 0,
    val prxFile: String = "",
    val prxName: String = ""
) : WithEmulator, BaseSceModule() {
    override val mmodule get() = this
    inline fun <reified T : SceModule> getModuleOrNull(): T? = emulator.moduleManager.modulesByClass[T::class] as? T?
    inline fun <reified T : SceModule> getModule(): T =
        getModuleOrNull<T>() ?: invalidOp1("Expected to get module ${T::class.portableSimpleName}")

    val loggerSuspend = Logger("SceModuleSuspend").apply {
        //level = LogLevel.TRACE
    }
    val logger = Logger("SceModule.$name").apply {
        //level = LogLevel.TRACE
    }

    fun registerPspModule() {
        registerModule()
    }

    open fun stopModule() {
    }

    protected abstract fun registerModule(): Unit

    private val rr: RegisterReader = RegisterReader()

    val functions = IntMap<NativeFunction>()

    fun registerFunctionRaw(function: NativeFunction) {
        functions[function.nid.toInt()] = function
        if (function.syscall >= 0) {
            emulator.syscalls.register(function.syscall, function.name) { cpu, syscall ->
                //println("REGISTERED SYSCALL $syscall")
                logger.trace { "${this.name}:${function.name}" }
                function.function(cpu)
            }
        }
    }

    fun registerFunctionRaw(
        name: String,
        uid: Long,
        since: Int = 150,
        syscall: Int = -1,
        function: (CpuState) -> Unit
    ) {
        registerFunctionRaw(NativeFunction(name, uid, since, syscall, function))
    }

    fun registerFunctionRR(
        name: String,
        uid: Long,
        since: Int = 150,
        syscall: Int = -1,
        function: RegisterReader.(CpuState) -> Unit
    ) {
        registerFunctionRaw(name, uid, since, syscall) {
            //when (name) {
            //	"sceGeListUpdateStallAddr", "sceKernelLibcGettimeofday" -> Unit
            //	else -> println("Calling $name")
            //}
            try {
                if (it._thread?.tracing == true) println("Calling $name from ${it._thread?.name}")
                rr.reset(it)
                function(rr, it)
            } catch (e: Throwable) {
                if (e !is EmulatorControlFlowException) {
                    Console.error("Error while processing '$name' :: at ${it.sPC.hex} :: $e")
                }
                throw e
            }
        }
    }

    fun registerFunctionVoid(
        name: String,
        uid: Long,
        since: Int = 150,
        syscall: Int = -1,
        function: RegisterReader.(CpuState) -> Unit
    ) {
        registerFunctionRR(name, uid, since, syscall, function)
    }

    fun registerFunctionInt(
        name: String,
        uid: Long,
        since: Int = 150,
        syscall: Int = -1,
        function: RegisterReader.(CpuState) -> Int
    ) {
        registerFunctionRR(name, uid, since, syscall) {
            this.cpu.r2 = try {
                function(it)
            } catch (e: SceKernelException) {
                e.errorCode
            }
        }
    }

    fun registerFunctionFloat(
        name: String,
        uid: Long,
        since: Int = 150,
        syscall: Int = -1,
        function: RegisterReader.(CpuState) -> Float
    ) {
        registerFunctionRR(name, uid, since, syscall) {
            this.cpu.setFpr(0, function(it))
        }
    }

    fun registerFunctionLong(
        name: String,
        uid: Long,
        since: Int = 150,
        syscall: Int = -1,
        function: RegisterReader.(CpuState) -> Long
    ) {
        registerFunctionRR(name, uid, since, syscall) {
            val ret = function(it)
            this.cpu.r2 = (ret ushr 0).toInt()
            this.cpu.r3 = (ret ushr 32).toInt()
        }
    }

    fun <T> registerFunctionSuspendT(
        name: String,
        uid: Long,
        since: Int = 150,
        syscall: Int = -1,
        cb: Boolean = false,
        function: suspend RegisterReader.(CpuState) -> T,
        resumeHandler: (CpuState, PspThread, T) -> Unit,
        convertErrorToT: (Int) -> T
    ) {
        val fullName = "${this.name}:$name"
        registerFunctionRR(name, uid, since, syscall) { rrr ->
            val rcpu = cpu
            val rthread = thread

            loggerSuspend.trace { "Suspend $name (${threadManager.summary}) : ${rcpu.summary}" }
            val mfunction: suspend (RegisterReader) -> T = { function(it, rcpu) }
            try {
                val result = mfunction.startCoroutineUninterceptedOrReturn(this, object : Continuation<T> {
                    override val context: CoroutineContext = coroutineContext

                    override fun resumeWith(result: Result<T>) {
                        if (result.isSuccess) {
                            val value = result.getOrThrow()
                            resumeHandler(rcpu, thread, value)
                            rthread.resume()
                            loggerSuspend.trace { "Resumed $name with value: $value (${threadManager.summary}) : ${rcpu.summary}" }
                        } else {
                            val e = result.exceptionOrNull()!!
                            when (e) {
                                is SceKernelException -> {
                                    resumeHandler(rcpu, thread, convertErrorToT(e.errorCode))
                                    rthread.resume()
                                }
                                else -> {
                                    println("ERROR at registerFunctionSuspendT.resumeWithException")
                                    e.printStackTrace()
                                    throw e
                                }
                            }
                        }
                    }
                })

                if (result == COROUTINE_SUSPENDED) {
                    rthread.markWaiting(WaitObject.COROUTINE(fullName), cb = cb)
                    if (rthread.tracing) println("  [S] Calling $name")
                    threadManager.suspendReturnVoid()
                } else {
                    resumeHandler(rthread.state, rthread, result as T)
                }
            } catch (e: SceKernelException) {
                resumeHandler(rthread.state, rthread, convertErrorToT(e.errorCode))
            } catch (e: CpuBreakException) {
                throw e
            } catch (e: Throwable) {
                println("ERROR at registerFunctionSuspendT.resumeWithException")
                e.printStackTrace()
                throw e
            }
        }
    }

    fun registerFunctionSuspendInt(
        name: String,
        uid: Long,
        since: Int = 150,
        syscall: Int = -1,
        cb: Boolean = false,
        function: suspend RegisterReader.(CpuState) -> Int
    ) {
        registerFunctionSuspendT<Int>(name, uid, since, syscall, cb, function,
            resumeHandler = { cpu, thread, value -> cpu.r2 = value },
            convertErrorToT = { it }
        )
    }

    fun registerFunctionSuspendLong(
        name: String,
        uid: Long,
        since: Int = 150,
        syscall: Int = -1,
        cb: Boolean = false,
        function: suspend RegisterReader.(CpuState) -> Long
    ) {
        registerFunctionSuspendT<Long>(name, uid, since, syscall, cb, function, resumeHandler = { cpu, thread, value ->
            cpu.r2 = (value ushr 0).toInt()
            cpu.r3 = (value ushr 32).toInt()
        }, convertErrorToT = { it.toLong() })
    }
}
