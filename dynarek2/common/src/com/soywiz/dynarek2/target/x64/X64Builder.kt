package com.soywiz.dynarek2.target.x64

// http://ref.x86asm.net/coder64.html
open class X64Builder : BaseBuilder() {
	fun retn() = bytes(0xc3)

    // MMX
    fun addss(dst: RegXmm, src: RegXmm) = bytes(0xF3, 0x0F, 0x58, 0xC0 or (dst.olindex shl 3) or (src.olindex shl 0))
    fun mulss(dst: RegXmm, src: RegXmm) = bytes(0xF3, 0x0F, 0x59, 0xC0 or (dst.olindex shl 3) or (src.olindex shl 0))
    fun subss(dst: RegXmm, src: RegXmm) = bytes(0xF3, 0x0F, 0x5C, 0xC0 or (dst.olindex shl 3) or (src.olindex shl 0))
    fun divss(dst: RegXmm, src: RegXmm) = bytes(0xF3, 0x0F, 0x5E, 0xC0 or (dst.olindex shl 3) or (src.olindex shl 0))
    fun readMem(reg: RegXmm, base: Reg64, offset: Int) =
        //bytes(0x67, 0xF3, 0x0F, 0x10, 0x80 or (reg.olindex shl 3) or ((base.olindex) shl 0)).also { int(offset) }
        bytes(0xF3, 0x0F, 0x10, 0x80 or (reg.olindex shl 3) or ((base.olindex) shl 0)).also { int(offset) }
    fun writeMem(reg: RegXmm, base: Reg64, offset: Int) =
        //bytes(0x67, 0xF3, 0x0F, 0x11, 0x80 or (reg.olindex shl 3) or ((base.olindex) shl 0)).also { int(offset) }
        bytes(0xF3, 0x0F, 0x11, 0x80 or (reg.olindex shl 3) or ((base.olindex) shl 0)).also { int(offset) }

    // MEM

    fun writeMem(reg: Reg8, base: Reg64, offset: Int) = bytes(0x88, 0x80 or (reg.olindex shl 3) or (base.olindex shl 0)).also { int(offset) }
    fun writeMem(reg: Reg16, base: Reg64, offset: Int) = bytes(0x66, 0x89, 0x80 or (reg.olindex shl 3) or (base.olindex shl 0)).also { int(offset) }
    fun writeMem(reg: Reg32, base: Reg64, offset: Int) = bytes(0x89, 0x80 or (reg.olindex shl 3) or (base.olindex shl 0)).also { int(offset) }

    fun readMem(reg: Reg8, base: Reg64, offset: Int) = bytes(0x8A, 0x80 or (reg.olindex shl 3) or (base.olindex shl 0)).also { int(offset) }
    fun readMem(reg: Reg16, base: Reg64, offset: Int) = bytes(0x66, 0x8B, 0x80 or (reg.olindex shl 3) or (base.olindex shl 0)).also { int(offset) }
    fun readMem(reg: Reg32, base: Reg64, offset: Int) = bytes(0x8B, 0x80 or (reg.olindex shl 3) or (base.olindex shl 0)).also { int(offset) }

    fun writeMem(reg: Reg64, base: Reg64, offset: Int) {
        bytes(if (reg.extended) 0x4C else 0x48, 0x89, 0x80 or (reg.lindex shl 3) or (base.lindex shl 0)).also { int(offset) }
    }
    fun readMem(reg: Reg64, base: Reg64, offset: Int) {
        bytes(if (reg.extended) 0x4C else 0x48, 0x8B, 0x80 or (reg.lindex shl 3) or (base.lindex shl 0)).also { int(offset) }
    }

    fun movEaxEbpOffset(offset: Int) {
		bytes(0x8b, 0x85)
		int(offset)
	}

    fun cpuid(kind: Int) {
        movEax(kind)
        cpuid()
    }

    fun cpuid() {
        bytes(0x0F, 0xA2)
    }

	fun xchg(a: Reg64, b: Reg64) {
		if (a == b) error("no effect")
		if (a.index > b.index) return xchg(b, a)
		if (a != Reg64.RAX) error("Only allow RAX to exchange")
		bytes(0x48, 0x90 or (b.index shl 0))
	}

	private fun _C0(dst: Reg32, src: Reg32) = 0b11_000_000 or (src.index shl 3) or (dst.index shl 0)
	private fun _C0(dst: Reg64, src: Reg64) = _C0(dst.to32(), src.to32())

	fun add(dst: Reg32, src: Reg32) = bytes(0x01, _C0(dst, src))
    fun or(dst: Reg32, src: Reg32) = bytes(0x09, _C0(dst, src))
    fun and(dst: Reg32, src: Reg32): Unit = bytes(0x21, _C0(dst, src))
    fun xor(dst: Reg32, src: Reg32): Unit = bytes(0x31, _C0(dst, src))
	fun sub(dst: Reg32, src: Reg32) = bytes(0x29, _C0(dst, src))

    val Int.fitsByte: Boolean get() = this.toByte().toInt() == this.toInt()

    fun sub(dst: Reg64, size: Int) {
        if (size.fitsByte) {
            bytes(0x48, 0x83, 0xE8 or (dst.olindex))
            bytes(size)
        } else {
            bytes(0x48, 0x81, 0xE8 or (dst.olindex))
            int(size)
        }
    }

    fun add(dst: Reg64, size: Int) {
        if (size.fitsByte) {
            bytes(0x48, 0x83, 0xC0 or (dst.olindex), size)
        } else {
            error("Doesn't fit byte")
        }
    }

    fun shl(dst: Reg32, src: Reg32): Unit = TODO()
    fun shr(dst: Reg32, src: Reg32): Unit = TODO()
    fun ushr(dst: Reg32, src: Reg32): Unit = TODO()

    fun shl(dst: Reg32, shift: Reg8): Unit {
        if (shift != Reg8.CL) error("Unsupported")
        bytes(0xD3, 0xE0 or dst.olindex)
    }

    fun shr(dst: Reg32, shift: Reg8): Unit {
        if (shift != Reg8.CL) error("Unsupported")
        bytes(0xD3, 0xF8 or dst.olindex)
    }

    fun ushr(dst: Reg32, shift: Reg8): Unit {
        if (shift != Reg8.CL) error("Unsupported")
        bytes(0xD3, 0xE8 or dst.olindex)
    }

    fun _mul(src: Reg32, signed: Boolean): Unit {
        if (src.extended) bytes(0x41)
        val sor = if (signed) 0x8 else 0
        bytes(0xf7, 0xe0 or sor or (src.lindex shl 0))
    }

    fun _div(src: Reg32, signed: Boolean): Unit {
        if (src.extended) bytes(0x49)
        val sor = if (signed) 0x8 else 0
        bytes(0xf7, 0xf0 or sor or (src.lindex shl 0))
    }

    fun mul(src: Reg32): Unit = _mul(src, signed = false)
    fun div(src: Reg32): Unit = _div(src, signed = false)
    fun imul(src: Reg32): Unit = _mul(src, signed = true)
    fun idiv(src: Reg32): Unit = _div(src, signed = true)

    fun mul(src: Reg64) = bytes(0x48).also { mul(Reg32(src.index)) }
	fun add(dst: Reg64, src: Reg64) = bytes(0x48).also { add(Reg32(dst.index), Reg32(src.index)) }



	fun mov(dst: Reg64, src: Reg64) {
        if (dst == src) return
		if (dst.index >= 8) TODO("Upper registers not implemented")
		bytes(0x48, 0x89, _C0(dst, src))
	}

	fun mov(dst: Reg32, value: Int) = bytes(0xb8 + dst.index).also { int(value) }

	fun push(r: Reg64) {
		if (r.index < 8) {
			bytes(0x50 + r.index)
		} else {
			bytes(0x41, 0x50 + r.index - 7)
		}
	}

	fun pushRAX() = push(Reg64.RAX)
	fun pushRBX() = push(Reg64.RBX)

	fun popRAX() = pop(Reg64.RAX)
	fun popRBX() = pop(Reg64.RBX)

	fun pop(r: Reg64) {
		if (r.index < 8) {
			bytes(0x58 + r.index)
		} else {
			bytes(0x41, 0x58 + r.index - 7)
		}
	}

    fun cmp(l: Reg32, r: Reg32) = bytes(0x39, 0xC0 or (l.olindex shl 0) or (r.olindex shl 3))
    fun cmp(l: Reg64, r: Reg64) = bytes(0x48).also { cmp(l.to32(), r.to32()) }

    fun cmp(l: Reg32, r: Int) {
        if (l == Reg32.EAX) {
            bytes(0x3D)
            int(r)
        } else {
            TODO()
        }
    }

    //fun jc(label: Label) = bytes(0x0F, 0x82).also { patch32(label) }

    fun je(label: Label) = bytes(0x0F, 0x84).also { patch32(label) }
    fun jne(label: Label) = bytes(0x0F, 0x85).also { patch32(label) }

    fun jl(label: Label) = bytes(0x0F, 0x8C).also { patch32(label) }
    fun jge(label: Label) = bytes(0x0F, 0x8D).also { patch32(label) }
    fun jle(label: Label) = bytes(0x0F, 0x8E).also { patch32(label) }
    fun jg(label: Label) = bytes(0x0F, 0x8F).also { patch32(label) }

    fun call(reg: Reg64) = bytes(0xFF, 0xD0 + reg.olindex)
    fun callAbsolute(value: Long) {
        mov(Reg64.RAX, value)
        call(Reg64.RAX)
    }
    fun call(label: Label) = bytes(0xe8).also { patch32(label) }
    fun jmp(label: Label) = bytes(0xe9).also { patch32(label) }

    fun patch32(label: Label) = patch(label, 4)

    fun nop() = bytes(0x90)

	fun movEax(value: Int) = mov(Reg32.EAX, value)
	fun movEdi(value: Int) = mov(Reg32.EDI, value)

	fun mov(reg: Reg64, value: Long) {
		bytes(0x48, 0xb8 or reg.olindex)
		long(value)
	}

	fun syscall() {
		bytes(0x0F, 0x05)
	}
}

open class BaseBuilder {
	private var pos = 0
	private var data = ByteArray(1024)

	fun getBytes(): ByteArray {
        patch()
        return data.copyOf(pos)
    }

    private fun patch() {
        val oldpos = pos
        try {
            for (patch in patches) {
                val label = patch.label
                val relative = (label.offset - patch.offset) - patch.incr
                pos = patch.offset
                int(relative)
            }
        } finally {
            pos = oldpos
        }
    }

    private val patches = arrayListOf<Patch>()

    fun patch(label: Label, incr: Int = 4, size: Int = incr) {
        patches += Patch(label, pos, incr)
        for (n in 0 until size) bytes(0)
    }

    fun place(label: Label) {
        label.offset = pos
    }

    class Patch(val label: Label, val offset: Int, val incr: Int)

    class Label {
        var offset: Int = -1
    }

	fun bytes(v: Int) {
		if (pos >= data.size) {
			data = data.copyOf(((data.size + 2) * 2))
		}
		data[pos++] = v.toByte()
	}

	fun bytes(a: Int, b: Int) = run { bytes(a); bytes(b) }
	fun bytes(a: Int, b: Int, c: Int) = run { bytes(a); bytes(b); bytes(c) }
	fun bytes(a: Int, b: Int, c: Int, d: Int) = run { bytes(a); bytes(b); bytes(c); bytes(d) }
    fun bytes(a: Int, b: Int, c: Int, d: Int, e: Int) = run { bytes(a); bytes(b); bytes(c); bytes(d); bytes(e) }

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

inline class Reg8(val index: Int) {
    companion object {
        val AL = Reg8(0)
        val CL = Reg8(1)
        val DL = Reg8(2)
        val BL = Reg8(3)
    }

    val extended get() = index >= 8
    val lindex get() = index and 0x7
    val olindex get() = if (extended) error("Unsupported extended") else lindex
}

inline class Reg16(val index: Int) {
    companion object {
        val AX = Reg16(0)
        val CX = Reg16(1)
        val DX = Reg16(2)
        val BX = Reg16(3)
        val SP = Reg16(4)
        val BP = Reg16(5)
        val SI = Reg16(6)
        val DI = Reg16(7)
    }

    val extended get() = index >= 8
    val lindex get() = index and 0x7
    val olindex get() = if (extended) error("Unsupported extended") else lindex

    fun to32() = Reg32(index)
    fun to64() = Reg64(index)
}

inline class Reg32(val index: Int) {
	companion object {
		val EAX = Reg32(0)
		val ECX = Reg32(1)
		val EDX = Reg32(2)
		val EBX = Reg32(3)
		val ESP = Reg32(4)
		val EBP = Reg32(5)
		val ESI = Reg32(6)
		val EDI = Reg32(7)

        val R8D = Reg32(8)
        val R9D = Reg32(9)
        val R10D = Reg32(10)
        val R11D = Reg32(11)
        val R12D = Reg32(12)
        val R13D = Reg32(13)
        val R14D = Reg32(14)
        val R15D = Reg32(15)

		val REGS = arrayOf(EAX, ECX, EDX, EBX)
	}

    val extended get() = index >= 8
    val lindex get() = index and 0x7
    val olindex get() = if (extended) error("Unsupported extended") else lindex
    fun to16() = Reg16(index)
	fun to64() = Reg64(index)
}

inline class Reg64(val index: Int) {
	companion object {
		val RAX = Reg64(0b0_000)
		val RCX = Reg64(0b0_001)
		val RDX = Reg64(0b0_010)
		val RBX = Reg64(0b0_011)
		val RSP = Reg64(0b0_100)
		val RBP = Reg64(0b0_101)
		val RSI = Reg64(0b0_110)
		val RDI = Reg64(0b0_111)

        // Extended
		val R8 = Reg64(0b1_000)
		val R9 = Reg64(0b1_001)
		val R10 = Reg64(0b1_010)
		val R11 = Reg64(0b1_011)
		val R12 = Reg64(0b1_100)
		val R13 = Reg64(0b1_101)
		val R14 = Reg64(0b1_110)
		val R15 = Reg64(0b1_111)
    }

    val extended get() = index >= 8
    val lindex get() = index and 0b111
    val olindex get() = if (extended) error("Unsupported extended") else lindex
    fun to16() = Reg16(index)
    fun to32() = Reg32(index)
}


inline class RegXmm(val index: Int) {
    companion object {
        val XMM0  = RegXmm(0)
        val XMM1  = RegXmm(1)
        val XMM2  = RegXmm(2)
        val XMM3  = RegXmm(3)
        val XMM4  = RegXmm(4)
        val XMM5  = RegXmm(5)
        val XMM6  = RegXmm(6)
        val XMM7  = RegXmm(7)
        val XMM8  = RegXmm(8)
        val XMM9  = RegXmm(9)
        val XMM10 = RegXmm(10)
        val XMM11 = RegXmm(11)
        val XMM12 = RegXmm(12)
        val XMM13 = RegXmm(13)
        val XMM14 = RegXmm(14)
        val XMM15 = RegXmm(15)
    }

    val extended get() = index >= 8
    val lindex get() = index and 0x7
    val olindex get() = if (extended) error("Unsupported extended") else lindex
    fun to16() = Reg16(index)
    fun to32() = Reg32(index)
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
