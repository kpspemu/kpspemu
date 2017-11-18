package com.soywiz.kpspemu.hle

import com.soywiz.kds.IntMap
import com.soywiz.klogger.Logger
import com.soywiz.korio.async.Promise
import com.soywiz.korio.coroutine.Continuation
import com.soywiz.korio.error.invalidOp
import com.soywiz.korio.lang.format
import com.soywiz.korio.lang.printStackTrace
import com.soywiz.korio.util.nextAlignedTo
import com.soywiz.kpspemu.Emulator
import com.soywiz.kpspemu.WithEmulator
import com.soywiz.kpspemu.coroutineContext
import com.soywiz.kpspemu.cpu.CpuState
import com.soywiz.kpspemu.hle.error.SceKernelException
import com.soywiz.kpspemu.hle.manager.PspThread
import com.soywiz.kpspemu.hle.manager.WaitObject
import com.soywiz.kpspemu.hle.manager.thread
import com.soywiz.kpspemu.mem.MemPtr
import com.soywiz.kpspemu.mem.Memory
import com.soywiz.kpspemu.mem.Ptr
import com.soywiz.kpspemu.mem.Ptr32
import com.soywiz.kpspemu.threadManager
import kotlin.coroutines.experimental.CoroutineContext
import kotlin.coroutines.experimental.startCoroutine

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
	val int: Int get() = this.cpu.getGpr(pos++)
	val long: Long
		get() {
			pos = pos.nextAlignedTo(2) // Ensure register alignment
			val low = this.cpu.getGpr(pos++)
			val high = this.cpu.getGpr(pos++)
			return (high.toLong() shl 32) or (low.toLong() and 0xFFFFFFFF)
		}
	val ptr: Ptr get() = MemPtr(mem, int)
	val ptr32: Ptr32 get() = Ptr32(ptr)
	val str: String? get() = mem.readStringzOrNull(int)
	val istr: String get() = mem.readStringzOrNull(int) ?: ""
}

data class NativeFunction(val name: String, val nid: Long, val since: Int, val syscall: Int, val function: (CpuState) -> Unit)

abstract class SceModule(
	override val emulator: Emulator,
	val name: String,
	val flags: Int = 0,
	val prxFile: String = "",
	val prxName: String = ""
) : WithEmulator {
	val logger = Logger("SceModule.$name")

	fun registerPspModule() {
		registerModule()
	}

	abstract protected fun registerModule(): Unit

	private val rr: RegisterReader = RegisterReader()

	val functions = IntMap<NativeFunction>()

	fun getByNidOrNull(nid: Int): NativeFunction? = functions[nid]
	fun getByNid(nid: Int): NativeFunction = getByNidOrNull(nid) ?: invalidOp("Can't find NID 0x%08X in %s".format(nid, name))

	fun UNIMPLEMENTED(nid: Int): Nothing {
		val func = getByNid(nid)
		TODO("Unimplemented %s:0x%08X:%s".format(this.name, func.nid, func.name))
	}

	fun UNIMPLEMENTED(nid: Long): Nothing = UNIMPLEMENTED(nid.toInt())

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

	fun registerFunctionRaw(name: String, uid: Long, since: Int = 150, syscall: Int = -1, function: (CpuState) -> Unit) {
		registerFunctionRaw(NativeFunction(name, uid, since, syscall, function))
	}

	fun registerFunctionRR(name: String, uid: Long, since: Int = 150, syscall: Int = -1, function: RegisterReader.(CpuState) -> Unit) {
		registerFunctionRaw(name, uid, since, syscall) {
			rr.reset(it)
			function(rr, it)
		}
	}

	fun registerFunctionVoid(name: String, uid: Long, since: Int = 150, syscall: Int = -1, function: RegisterReader.(CpuState) -> Unit) {
		registerFunctionRR(name, uid, since, syscall, function)
	}

	fun registerFunctionInt(name: String, uid: Long, since: Int = 150, syscall: Int = -1, function: RegisterReader.(CpuState) -> Int) {
		registerFunctionRR(name, uid, since, syscall) {
			this.cpu.r2 = try {
				function(it)
			} catch (e: SceKernelException) {
				e.errorCode
			}
		}
	}

	fun registerFunctionFloat(name: String, uid: Long, since: Int = 150, syscall: Int = -1, function: RegisterReader.(CpuState) -> Float) {
		registerFunctionRR(name, uid, since, syscall) {
			this.cpu.setFpr(0, function(it))
		}
	}

	fun registerFunctionLong(name: String, uid: Long, since: Int = 150, syscall: Int = -1, function: RegisterReader.(CpuState) -> Long) {
		registerFunctionRR(name, uid, since, syscall) {
			val ret = function(it)
			this.cpu.r2 = (ret ushr 0).toInt()
			this.cpu.r3 = (ret ushr 32).toInt()
		}
	}

	// @TODO: registerFunctionSuspendInt and registerFunctionSuspendLong are mostly duplicated!

	fun registerFunctionSuspendInt(name: String, uid: Long, since: Int = 150, syscall: Int = -1, cb: Boolean = false, function: suspend RegisterReader.(CpuState) -> Int) {
		val fullName = "${this.name}:$name"
		registerFunctionRR(name, uid, since, syscall) {
			val mfunction: suspend (RegisterReader) -> Int = { function(it, it.cpu) }
			var completed = false
			mfunction.startCoroutine(this, object : Continuation<Int> {
				override val context: CoroutineContext = coroutineContext

				override fun resume(value: Int) {
					logger.trace { "Resumed $name with value: $value" }
					cpu.r2 = value
					completed = true
					it.thread.resume()
				}

				override fun resumeWithException(exception: Throwable) {
					if (exception is SceKernelException) {
						resume(exception.errorCode)
					} else {
						exception.printStackTrace()
						throw exception
					}
				}
			})

			if (!completed) {
				it.thread.markWaiting(WaitObject.PROMISE(Promise(), fullName), cb = cb)
				threadManager.suspend()
			}
		}
	}

	fun registerFunctionSuspendLong(name: String, uid: Long, since: Int = 150, syscall: Int = -1, cb: Boolean = false, function: suspend RegisterReader.(CpuState) -> Long) {
		val fullName = "${this.name}:$name"
		registerFunctionRR(name, uid, since, syscall) {
			val mfunction: suspend (RegisterReader) -> Long = { function(it, it.cpu) }
			var completed = false
			mfunction.startCoroutine(this, object : Continuation<Long> {
				override val context: CoroutineContext = coroutineContext

				override fun resume(value: Long) {
					cpu.r2 = (value ushr 0).toInt()
					cpu.r3 = (value ushr 32).toInt()
					completed = true
					it.thread.resume()
				}

				override fun resumeWithException(exception: Throwable) {
					exception.printStackTrace()
					throw exception
				}
			})

			if (!completed) {
				it.thread.markWaiting(WaitObject.PROMISE(Promise(), fullName), cb = cb)
				threadManager.suspend()
			}
		}
	}
}
