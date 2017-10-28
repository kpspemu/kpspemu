package com.soywiz.kpspemu.hle.modules

import com.soywiz.korio.ds.lmapOf
import com.soywiz.kpspemu.Emulator
import com.soywiz.kpspemu.cpu.CpuState
import com.soywiz.kpspemu.mem.Memory

class RegisterReader {
	var pos: Int = 4
	lateinit var cpu: CpuState

	fun reset(cpu: CpuState) {
		this.cpu = cpu
		this.pos = 4
	}

	val mem: Memory get() = cpu.mem
	val int: Int get() = this.cpu.GPR[pos++]
}

open class SceModule {
	private lateinit var e: Emulator

	fun registerPspModule(e: Emulator) {
		this.e = e
		registerModule()
	}

	open protected fun registerModule() {
	}

	private val rr: RegisterReader = RegisterReader()

	data class ModuleFunction(val name: String, val uid: Long, val since: Int, val syscall: Int, val function: (CpuState) -> Unit)

	val functions = lmapOf<Int, ModuleFunction>()

	protected fun registerFunctionRaw(function: ModuleFunction) {
		functions[function.uid.toInt()] = function
		if (function.syscall >= 0) {
			e.syscalls.register(function.syscall) { cpu, syscall ->
				//println("REGISTERED SYSCALL $syscall")
				function.function(cpu)
			}
		}
	}

	protected fun registerFunctionRaw(name: String, uid: Long, since: Int = 150, syscall: Int = -1, function: (CpuState) -> Unit) {
		registerFunctionRaw(ModuleFunction(name, uid, since, syscall, function))
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
}