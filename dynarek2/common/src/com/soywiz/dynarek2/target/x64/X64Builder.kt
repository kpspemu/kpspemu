package com.soywiz.dynarek2.target.x64

open class X64Builder : BaseX64Builder() {
	fun retn() {
		bytes(0xc3)
	}

	fun movEaxEbpOffset(offset: Int) {
		bytes(0x8b, 0x85)
		int(offset)
	}

	fun xchg(a: X64Reg64, b: X64Reg64) {
		if (a == b) error("no effect")
		if (a.index > b.index) return xchg(b, a)
		if (a != X64Reg64.RAX) error("Only allow RAX to exchange")
		bytes(0x48, 0x90 or (b.index shl 0))
	}

	private fun _C0(dst: X64Reg32, src: X64Reg32) = 0b11_000_000 or (src.index shl 3) or (dst.index shl 0)
	private fun _C0(dst: X64Reg64, src: X64Reg64) = _C0(dst.to32(), src.to32())

	fun add(dst: X64Reg32, src: X64Reg32) = bytes(0x01, _C0(dst, src))
	fun sub(dst: X64Reg32, src: X64Reg32) = bytes(0x29, _C0(dst, src))
	fun mul(src: X64Reg32) = bytes(0xf7, 0xe0 or (src.index shl 0))

	fun mov(dst: X64Reg64, src: X64Reg64) {
		if (dst.index >= 8) TODO("Upper registers not implemented")
		bytes(0x48, 0x89, _C0(dst, src))
	}

	fun mov(dst: X64Reg32, value: Int) {
		when (dst) {
			X64Reg32.EAX -> movEax(value)
			else -> TODO("mov dst, value")
		}
	}

	fun push(r: X64Reg64) {
		if (r.index < 8) {
			bytes(0x50 + r.index)
		} else {
			bytes(0x41, 0x50 + r.index - 7)
		}
	}

	fun pushRAX() = push(X64Reg64.RAX)
	fun pushRBX() = push(X64Reg64.RBX)

	fun popRAX() = pop(X64Reg64.RAX)
	fun popRBX() = pop(X64Reg64.RBX)

	fun pop(r: X64Reg64) {
		if (r.index < 8) {
			bytes(0x58 + r.index)
		} else {
			bytes(0x41, 0x58 + r.index - 7)
		}
	}

	fun movEax(value: Int) {
		bytes(0xb8)
		int(value)
	}

	fun movEdi(value: Int) {
		bytes(0xbf)
		int(value)
	}

	fun movRax(value: Long) {
		bytes(0x48, 0xb0)
		long(value)
	}

	fun syscall() {
		bytes(0x0F, 0x05)
	}
}

open class BaseX64Builder {
	var pos = 0
	var data = ByteArray(1024)

	fun getBytes() = data.copyOf(pos)

	fun bytes(v: Int) {
		if (pos >= data.size) {
			data = data.copyOf(((data.size + 2) * 2))
		}
		data[pos++] = v.toByte()
	}

	fun bytes(a: Int, b: Int) = run { bytes(a); bytes(b) }
	fun bytes(a: Int, b: Int, c: Int) = run { bytes(a); bytes(b); bytes(c) }
	fun bytes(a: Int, b: Int, c: Int, d: Int) = run { bytes(a); bytes(b); bytes(c); bytes(d) }

	fun short(v: Int) {
		bytes((v ushr 0) and 0xFF, (v ushr 8) and 0xFF)
	}

	fun int(v: Int) {
		short((v ushr 0) and 0xFFFF)
		short((v ushr 16) and 0xFFFF)
	}

	fun long(v: Long) {
		int((v ushr 0).toInt())
		int((v ushr 32).toInt())
	}
}

inline class X64Reg32(val index: Int) {
	companion object {
		val EAX = X64Reg32(0b000)
		val ECX = X64Reg32(0b001)
		val EDX = X64Reg32(0b010)
		val EBX = X64Reg32(0b011)
		val ESP = X64Reg32(0b100)
		val EBP = X64Reg32(0b101)
		val ESI = X64Reg32(0b110)
		val EDI = X64Reg32(0b111)

		val REGS = arrayOf(EAX, ECX, EDX, EBX)
	}

	fun to64() = X64Reg64(index)
}

inline class X64Reg64(val index: Int) {
	companion object {
		val RAX = X64Reg64(0b0_000)
		val RCX = X64Reg64(0b0_001)
		val RDX = X64Reg64(0b0_010)
		val RBX = X64Reg64(0b0_011)
		val RSP = X64Reg64(0b0_100)
		val RBP = X64Reg64(0b0_101)
		val RSI = X64Reg64(0b0_110)
		val RDI = X64Reg64(0b0_111)

		val R8 = X64Reg64(0b1_000)
		val R9 = X64Reg64(0b1_001)
		val R10 = X64Reg64(0b1_010)
		val R11 = X64Reg64(0b1_011)
		val R12 = X64Reg64(0b1_100)
		val R13 = X64Reg64(0b1_101)
		val R14 = X64Reg64(0b1_110)
		val R15 = X64Reg64(0b1_111)
	}

	fun to32() = X64Reg32(index)
}

/*
import kotlinx.cinterop.*
import platform.posix.*

fun main(args: Array<String>) {
    val code = ExecutableData(byteArrayOf(0xb8.toByte(), 0x07, 0, 0, 0, 0xc3.toByte())) // MOV EAX, 1; RETN
    val func = code.ptr.reinterpret<CFunction<() -> Int>>()
    println("Hello: " + func())
}

class ExecutableData(data: ByteArray) {
    val len = data.size
    //val len = getpagesize()
    val ptr = mmap(
            null, len.toLong(),
            PROT_READ or PROT_WRITE or PROT_EXEC,
            MAP_ANONYMOUS or MAP_SHARED, -1, 0
    )?.reinterpret<ByteVar>() ?: error("Couldn't reserve memory")

    init {
        for (n in 0 until data.size) ptr[n] = data[n]
    }

    fun free() {
        munmap(ptr, len.toLong())
    }
}
 */

/*

// https://msdn.microsoft.com/en-us/library/9z1stfyw.aspx

Register	Status	Use
RAX	Volatile	Return value register
RCX	Volatile	First integer argument
RDX	Volatile	Second integer argument
R8	Volatile	Third integer argument
R9	Volatile	Fourth integer argument
R10:R11	Volatile	Must be preserved as needed by caller; used in syscall/sysret instructions
R12:R15	Nonvolatile	Must be preserved by callee
RDI	Nonvolatile	Must be preserved by callee
RSI	Nonvolatile	Must be preserved by callee
RBX	Nonvolatile	Must be preserved by callee
RBP	Nonvolatile	May be used as a frame pointer; must be preserved by callee
RSP	Nonvolatile	Stack pointer
XMM0, YMM0	Volatile	First FP argument; first vector-type argument when __vectorcall is used
XMM1, YMM1	Volatile	Second FP argument; second vector-type argument when __vectorcall is used
XMM2, YMM2	Volatile	Third FP argument; third vector-type argument when __vectorcall is used
XMM3, YMM3	Volatile	Fourth FP argument; fourth vector-type argument when __vectorcall is used
XMM4, YMM4	Volatile	Must be preserved as needed by caller; fifth vector-type argument when __vectorcall is used
XMM5, YMM5	Volatile	Must be preserved as needed by caller; sixth vector-type argument when __vectorcall is used
XMM6:XMM15, YMM6:YMM15	Nonvolatile (XMM), Volatile (upper half of YMM)	Must be preserved as needed by callee. YMM registers must be preserved as needed by caller.
 */