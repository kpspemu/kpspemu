package com.soywiz.kpspemu.hle

import com.soywiz.korio.error.invalidOp
import com.soywiz.korio.lang.format
import com.soywiz.kpspemu.Emulator
import com.soywiz.kpspemu.WithEmulator
import com.soywiz.kpspemu.cpu.CpuState
import com.soywiz.kpspemu.hle.manager.PspThread
import com.soywiz.kpspemu.hle.manager.thread
import com.soywiz.kpspemu.mem.MemPtr
import com.soywiz.kpspemu.mem.Memory
import com.soywiz.kpspemu.mem.Ptr

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
	val int: Int get() = this.cpu.GPR[pos++]
	val ptr: Ptr get() = MemPtr(mem, int)
	val str: String? get() = mem.readStringzOrNull(int)
}

data class NativeFunction(val name: String, val nid: Long, val since: Int, val syscall: Int, val function: (CpuState) -> Unit)

abstract class SceModule(
	override val emulator: Emulator,
	val name: String,
	val flags: Int = 0,
	val prxFile: String = "",
	val prxName: String = ""
) : WithEmulator {
	fun registerPspModule() {
		registerModule()
	}

	abstract protected fun registerModule(): Unit

	private val rr: RegisterReader = RegisterReader()

	val functions = LinkedHashMap<Int, NativeFunction>()

	fun getByNidOrNull(nid: Int): NativeFunction? = functions[nid]
	fun getByNid(nid: Int): NativeFunction = getByNidOrNull(nid) ?: invalidOp("Can't find NID 0x%08X in %s".format(nid, name))

	fun UNIMPLEMENTED(nid: Int): Nothing {
		val func = getByNid(nid)
		TODO("Unimplemented %s:0x%08X:%s".format(this.name, func.nid, func.name))
	}

	fun UNIMPLEMENTED(nid: Long): Nothing = UNIMPLEMENTED(nid.toInt())

	protected fun registerFunctionRaw(function: NativeFunction) {
		functions[function.nid.toInt()] = function
		if (function.syscall >= 0) {
			emulator.syscalls.register(function.syscall) { cpu, syscall ->
				//println("REGISTERED SYSCALL $syscall")
				function.function(cpu)
			}
		}
	}

	protected fun registerFunctionRaw(name: String, uid: Long, since: Int = 150, syscall: Int = -1, function: (CpuState) -> Unit) {
		registerFunctionRaw(NativeFunction(name, uid, since, syscall, function))
	}

	protected fun registerFunctionRR(name: String, uid: Long, since: Int = 150, syscall: Int = -1, function: RegisterReader.(CpuState) -> Unit) {
		registerFunctionRaw(name, uid, since, syscall) {
			rr.reset(it)
			function(rr, it)
		}
	}

	protected fun registerFunctionVoid(name: String, uid: Long, since: Int = 150, syscall: Int = -1, function: RegisterReader.(CpuState) -> Unit) {
		registerFunctionRR(name, uid, since, syscall, function)
	}

	protected fun registerFunctionInt(name: String, uid: Long, since: Int = 150, syscall: Int = -1, function: RegisterReader.(CpuState) -> Int) {
		registerFunctionRR(name, uid, since, syscall) {
			this.cpu.r2 = function(it)
		}
	}

	protected fun registerFunctionLong(name: String, uid: Long, since: Int = 150, syscall: Int = -1, function: RegisterReader.(CpuState) -> Long) {
		registerFunctionRR(name, uid, since, syscall) {
			val ret = function(it)
			this.cpu.r2 = (ret ushr 0).toInt()
			this.cpu.r3 = (ret ushr 32).toInt()
		}
	}
}
