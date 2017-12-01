package com.soywiz.kpspemu.cpu

import com.soywiz.dynarek.JvmField
import com.soywiz.kds.Extra
import com.soywiz.kmem.*
import com.soywiz.korio.error.invalidArg
import com.soywiz.korio.lang.format
import com.soywiz.korio.util.extract
import com.soywiz.korio.util.insert
import com.soywiz.kpspemu.mem.DummyMemory
import com.soywiz.kpspemu.mem.Memory

data class CpuBreakException(val id: Int) : Exception() {
	companion object {
		val THREAD_WAIT = 10001
		val THREAD_EXIT_KILL = 10002
		val INTERRUPT_RETURN = 10003

		val THREAD_WAIT_RA = Memory.MAIN_OFFSET + 0
		val THREAD_EXIT_KIL_RA = Memory.MAIN_OFFSET + 4
		val INTERRUPT_RETURN_RA = Memory.MAIN_OFFSET + 8

		fun initialize(mem: Memory) {
			mem.sw(THREAD_WAIT_RA, 0b000000_00000000000000000000_001101 or (CpuBreakException.THREAD_WAIT shl 6))
			mem.sw(THREAD_EXIT_KIL_RA, 0b000000_00000000000000000000_001101 or (CpuBreakException.THREAD_EXIT_KILL shl 6))
			mem.sw(INTERRUPT_RETURN_RA, 0b000000_00000000000000000000_001101 or (CpuBreakException.INTERRUPT_RETURN shl 6))
		}
	}
}

// http://www.cs.uwm.edu/classes/cs315/Bacon/Lecture/HTML/ch05s03.html
var CpuState.K0: Int; set(value) = run { r26 = value }; get() = r26
var CpuState.K1: Int; set(value) = run { r27 = value }; get() = r27
var CpuState.GP: Int; set(value) = run { r28 = value }; get() = r28
var CpuState.SP: Int; set(value) = run { r29 = value }; get() = r29
var CpuState.FP: Int; set(value) = run { r30 = value }; get() = r30
var CpuState.RA: Int; set(value) = run { r31 = value }; get() = r31

var CpuState.V0: Int; set(value) = run { r2 = value }; get() = r2
var CpuState.V1: Int; set(value) = run { r3 = value }; get() = r3
var CpuState.A0: Int; set(value) = run { r4 = value }; get() = r4
var CpuState.A1: Int; set(value) = run { r5 = value }; get() = r5
var CpuState.A2: Int; set(value) = run { r6 = value }; get() = r6
var CpuState.A3: Int; set(value) = run { r7 = value }; get() = r7

data class RegInfo(val index: Int, val name: String, val mnemonic: String, val desc: String)

class CpuState(val globalCpuState: GlobalCpuState, val mem: Memory, val syscalls: Syscalls = TraceSyscallHandler()) : Extra by Extra.Mixin() {
	companion object {
		val gprInfos = listOf(
			RegInfo( 0, "r0", "zero", "Permanently 0"),
			RegInfo( 1, "r1", "at", "Assembler Temporaty"),
			RegInfo( 2, "r2", "v0", "Value returned by a subroutine"),
			RegInfo( 3, "r3", "v1", "Value returned by a subroutine"),
			RegInfo( 4, "r4", "a0", "Subroutine Arguments"),
			RegInfo( 5, "r5", "a1", "Subroutine Arguments"),
			RegInfo( 6, "r6", "a2", "Subroutine Arguments"),
			RegInfo( 7, "r7", "a3", "Subroutine Arguments"),
			RegInfo( 8, "r8", "t0", "Temporary"),
			RegInfo( 9, "r9", "t1", "Temporary"),
			RegInfo(10, "r10", "t2", "Temporary"),
			RegInfo(11, "r11", "t3", "Temporary"),
			RegInfo(12, "r12", "t4", "Temporary"),
			RegInfo(13, "r13", "t5", "Temporary"),
			RegInfo(14, "r14", "t6", "Temporary"),
			RegInfo(15, "r15", "t7", "Temporary"),
			RegInfo(16, "r16", "s0", "Saved registers"),
			RegInfo(17, "r17", "s1", "Saved registers"),
			RegInfo(18, "r18", "s2", "Saved registers"),
			RegInfo(19, "r19", "s3", "Saved registers"),
			RegInfo(20, "r20", "s4", "Saved registers"),
			RegInfo(21, "r21", "s5", "Saved registers"),
			RegInfo(22, "r22", "s6", "Saved registers"),
			RegInfo(23, "r23", "s7", "Saved registers"),
			RegInfo(24, "r24", "t8", "Temporary"),
			RegInfo(25, "r25", "t9", "Temporary"),
			RegInfo(26, "r26", "k0", "Kernel"),
			RegInfo(27, "r27", "k1", "Kernel"),
			RegInfo(28, "r28", "gp", "Global Pointer"),
			RegInfo(29, "r29", "sp", "Stack Pointer"),
			RegInfo(30, "r30", "fp", "Frame Pointer"),
			RegInfo(31, "r31", "fp", "Return Address")
		)

		val gprInfosByMnemonic = (gprInfos.map { it.mnemonic to it } + gprInfos.map { it.name to it }).toMap()

		val dummy = CpuState(GlobalCpuState.dummy, DummyMemory)

		var lastId = 0


		fun getGprProp(index: Int) = when (index) {
			0 -> CpuState::r0;1 -> CpuState::r1;2 -> CpuState::r2;3 -> CpuState::r3;4 -> CpuState::r4;5 -> CpuState::r5;6 -> CpuState::r6;7 -> CpuState::r7;
			8 -> CpuState::r8;9 -> CpuState::r9;10 -> CpuState::r10;11 -> CpuState::r11;12 -> CpuState::r12;13 -> CpuState::r13;14 -> CpuState::r14;15 -> CpuState::r15;
			16 -> CpuState::r16;17 -> CpuState::r17;18 -> CpuState::r18;19 -> CpuState::r19;20 -> CpuState::r20;21 -> CpuState::r21;22 -> CpuState::r22;23 -> CpuState::r23;
			24 -> CpuState::r24;25 -> CpuState::r25;26 -> CpuState::r26;27 -> CpuState::r27;28 -> CpuState::r28;29 -> CpuState::r29;30 -> CpuState::r30;31 -> CpuState::r31;
			else -> invalidArg("Invalid register $index")
		}
	}

	fun getGprProp(index: Int) = when (index) {
		0 -> ::r0;1 -> ::r1;2 -> ::r2;3 -> ::r3;4 -> ::r4;5 -> ::r5;6 -> ::r6;7 -> ::r7;
		8 -> ::r8;9 -> ::r9;10 -> ::r10;11 -> ::r11;12 -> ::r12;13 -> ::r13;14 -> ::r14;15 -> ::r15;
		16 -> ::r16;17 -> ::r17;18 -> ::r18;19 -> ::r19;20 -> ::r20;21 -> ::r21;22 -> ::r22;23 -> ::r23;
		24 -> ::r24;25 -> ::r25;26 -> ::r26;27 -> ::r27;28 -> ::r28;29 -> ::r29;30 -> ::r30;31 -> ::r31;
		else -> invalidArg("Invalid register $index")
	}

	val id = lastId++
	var totalExecuted: Long = 0L

	val _FMem = MemBufferAlloc(32 * 4)
	var _F = _FMem.asFloat32Buffer()
	var _FI = _FMem.asInt32Buffer()

	val _VFPRMem = MemBufferAlloc(128 * 4)
	val _VFPR = _VFPRMem.asFloat32Buffer()
	val _VFPR_I = _VFPRMem.asInt32Buffer()

	var fcr0: Int = 0x00003351
	var fcr25: Int = 0
	var fcr26: Int = 0
	var fcr27: Int = 0
	var fcr28: Int = 0
	var fcr31: Int = 0x00000e00

	var vpfxsEnabled = false
	var vpfxtEnabled = false
	var vpfxdEnabled = false
	var vpfxs: Int = 0xDC0000E4.toInt()
	var vpfxt: Int = 0xDC0000E4.toInt()
	var vpfxd: Int = 0x00000000

	fun updateFCR31(value: Int) {
		fcr31 = value and 0x0183FFFF
	}

	var fcr31_rm: Int set(value) = run { fcr31 = fcr31.insert(value, 0, 2) }; get() = fcr31.extract(0, 2)
	var fcr31_2_21: Int set(value) = run { fcr31 = fcr31.insert(value, 2, 21) }; get() = fcr31.extract(2, 21)
	var fcr31_cc: Boolean set(value) = run { fcr31 = fcr31.insert(value, 23) }; get() = fcr31.extract(23)
	var fcr31_fs: Boolean set(value) = run { fcr31 = fcr31.insert(value, 24) }; get() = fcr31.extract(24)
	var fcr31_25_7: Int set(value) = run { fcr31 = fcr31.insert(value, 25, 7) }; get() = fcr31.extract(25, 7)

	// @TODO: Fast version for dynarek

	//@JvmField @JsName("r0") var r0: Int = 0
	//@JvmField @JsName("r1") var r1: Int = 0
	//@JvmField @JsName("r2") var r2: Int = 0
	//@JvmField @JsName("r3") var r3: Int = 0
	//@JvmField @JsName("r4") var r4: Int = 0
	//@JvmField @JsName("r5") var r5: Int = 0
	//@JvmField @JsName("r6") var r6: Int = 0
	//@JvmField @JsName("r7") var r7: Int = 0
	//@JvmField @JsName("r8") var r8: Int = 0
	//@JvmField @JsName("r9") var r9: Int = 0
	//@JvmField @JsName("r10") var r10: Int = 0
	//@JvmField @JsName("r11") var r11: Int = 0
	//@JvmField @JsName("r12") var r12: Int = 0
	//@JvmField @JsName("r13") var r13: Int = 0
	//@JvmField @JsName("r14") var r14: Int = 0
	//@JvmField @JsName("r15") var r15: Int = 0
	//@JvmField @JsName("r16") var r16: Int = 0
	//@JvmField @JsName("r17") var r17: Int = 0
	//@JvmField @JsName("r18") var r18: Int = 0
	//@JvmField @JsName("r19") var r19: Int = 0
	//@JvmField @JsName("r20") var r20: Int = 0
	//@JvmField @JsName("r21") var r21: Int = 0
	//@JvmField @JsName("r22") var r22: Int = 0
	//@JvmField @JsName("r23") var r23: Int = 0
	//@JvmField @JsName("r24") var r24: Int = 0
	//@JvmField @JsName("r25") var r25: Int = 0
	//@JvmField @JsName("r26") var r26: Int = 0
	//@JvmField @JsName("r27") var r27: Int = 0
	//@JvmField @JsName("r28") var r28: Int = 0
	//@JvmField @JsName("r29") var r29: Int = 0
	//@JvmField @JsName("r30") var r30: Int = 0
	//@JvmField @JsName("r31") var r31: Int = 0
//
	//fun getGpr(index: Int): Int {
	//	return when (index) {
	//		0 -> 0;1 -> r1;2 -> r2;3 -> r3;4 -> r4;5 -> r5;6 -> r6;7 -> r7;
	//		8 -> r8;9 -> r9;10 -> r10;11 -> r11;12 -> r12;13 -> r13;14 -> r14;15 -> r15;
	//		16 -> r16;17 -> r17;18 -> r18;19 -> r19;20 -> r20;21 -> r21;22 -> r22;23 -> r23;
	//		24 -> r24;25 -> r25;26 -> r26;27 -> r27;28 -> r28;29 -> r29;30 -> r30;31 -> r31;
	//		else -> invalidOp
	//	}
	//}
//
	//fun setGpr(index: Int, v: Int): Unit {
	//	when (index) {
	//		0 -> Unit;1 -> r1 = v;2 -> r2 = v;3 -> r3 = v;4 -> r4 = v;5 -> r5 = v;6 -> r6 = v;7 -> r7 = v;
	//		8 -> r8 = v;9 -> r9 = v;10 -> r10 = v;11 -> r11 = v;12 -> r12 = v;13 -> r13 = v;14 -> r14 = v;15 -> r15 = v;
	//		16 -> r16 = v;17 -> r17 = v;18 -> r18 = v;19 -> r19 = v;20 -> r20 = v;21 -> r21 = v;22 -> r22 = v;23 -> r23 = v;
	//		24 -> r24 = v;25 -> r25 = v;26 -> r26 = v;27 -> r27 = v;28 -> r28 = v;29 -> r29 = v;30 -> r30 = v;31 -> r31 = v;
	//		else -> invalidOp
	//	}
	//}

	// @TODO: Fast version for interpreted

	var _R = IntArray(32)
	var r0: Int; set(value) = Unit; get() = 0
	var r1: Int; set(value) = run { _R[1] = value }; get() = _R[1]
	var r2: Int; set(value) = run { _R[2] = value }; get() = _R[2]
	var r3: Int; set(value) = run { _R[3] = value }; get() = _R[3]
	var r4: Int; set(value) = run { _R[4] = value }; get() = _R[4]
	var r5: Int; set(value) = run { _R[5] = value }; get() = _R[5]
	var r6: Int; set(value) = run { _R[6] = value }; get() = _R[6]
	var r7: Int; set(value) = run { _R[7] = value }; get() = _R[7]
	var r8: Int; set(value) = run { _R[8] = value }; get() = _R[8]
	var r9: Int; set(value) = run { _R[9] = value }; get() = _R[9]
	var r10: Int; set(value) = run { _R[10] = value }; get() = _R[10]
	var r11: Int; set(value) = run { _R[11] = value }; get() = _R[11]
	var r12: Int; set(value) = run { _R[12] = value }; get() = _R[12]
	var r13: Int; set(value) = run { _R[13] = value }; get() = _R[13]
	var r14: Int; set(value) = run { _R[14] = value }; get() = _R[14]
	var r15: Int; set(value) = run { _R[15] = value }; get() = _R[15]
	var r16: Int; set(value) = run { _R[16] = value }; get() = _R[16]
	var r17: Int; set(value) = run { _R[17] = value }; get() = _R[17]
	var r18: Int; set(value) = run { _R[18] = value }; get() = _R[18]
	var r19: Int; set(value) = run { _R[19] = value }; get() = _R[19]
	var r20: Int; set(value) = run { _R[20] = value }; get() = _R[20]
	var r21: Int; set(value) = run { _R[21] = value }; get() = _R[21]
	var r22: Int; set(value) = run { _R[22] = value }; get() = _R[22]
	var r23: Int; set(value) = run { _R[23] = value }; get() = _R[23]
	var r24: Int; set(value) = run { _R[24] = value }; get() = _R[24]
	var r25: Int; set(value) = run { _R[25] = value }; get() = _R[25]
	var r26: Int; set(value) = run { _R[26] = value }; get() = _R[26]
	var r27: Int; set(value) = run { _R[27] = value }; get() = _R[27]
	var r28: Int; set(value) = run { _R[28] = value }; get() = _R[28]
	var r29: Int; set(value) = run { _R[29] = value }; get() = _R[29]
	var r30: Int; set(value) = run { _R[30] = value }; get() = _R[30]
	var r31: Int; set(value) = run { _R[31] = value }; get() = _R[31]
	fun getGpr(index: Int): Int = _R[index]
	fun setGpr(index: Int, v: Int): Unit = run { if (index != 0) _R[index] = v }

	fun writeRegisters(addr: Int, start: Int = 0, count: Int = 32 - start) {
		for (n in 0 until count) mem.sw(addr + n * 4, getGpr(start + n))
	}

	fun readRegisters(addr: Int, start: Int = 0, count: Int = 32 - start) {
		for (n in 0 until count) setGpr(start + n, mem.lw(addr + n * 4))
	}

	//val FPR = FloatArray(32) { 0f }
	//val FPR_I = FprI(this)

	@JvmField var IR: Int = 0
	@JvmField var _PC: Int = 0
	@JvmField var _nPC: Int = 0
	@JvmField var LO: Int = 0
	@JvmField var HI: Int = 0
	@JvmField var IC: Int = 0

	fun getPCRef() = ::sPC

	var sPC get() = PC; set(value) = run { setPC(value) }

	val PC: Int get() = _PC

	var HI_LO: Long
		get() = (HI.toLong() shl 32) or (LO.toLong() and 0xFFFFFFFF)
		set(value) {
			HI = (value ushr 32).toInt()
			LO = (value ushr 0).toInt()
		}

	inline fun setPC(pc: Int) {
		_PC = pc
		_nPC = pc + 4
	}

	inline fun jump(pc: Int) {
		_PC = pc
		_nPC = pc + 4
	}

	inline fun advance_pc(offset: Int) {
		_PC = _nPC
		_nPC += offset
	}

	//fun getGpr(index: Int): Int = _R[index and 0x1F]
	//fun setGpr(index: Int, v: Int): Unit = run { if (index != 0) _R[index and 0x1F] = v }

	fun getFpr(index: Int): Float = _F[index]
	fun setFpr(index: Int, v: Float): Unit = run { _F[index] = v }
	fun getFprI(index: Int): Int = _FI[index]
	fun setFprI(index: Int, v: Int): Unit = run { _FI[index] = v }

	fun setVfpr(index: Int, value: Float) = run { _VFPR[index] = value }
	fun getVfpr(index: Int): Float = _VFPR[index]

	fun setVfprI(index: Int, value: Int) = run { _VFPR_I[index] = value }
	fun getVfprI(index: Int): Int = _VFPR_I[index]

	fun syscall(syscall: Int): Unit = syscalls.syscall(this, syscall)

	fun clone() = CpuState(globalCpuState, mem, syscalls).apply {
		this@CpuState.copyTo(this)
	}

	fun setTo(src: CpuState) = run { src.copyTo(this) }

	fun copyTo(dst: CpuState) {
		val src = this
		dst._PC = src._PC
		dst._nPC = src._nPC
		dst.HI = src.HI
		dst.LO = src.LO
		dst.IC = src.IC
		dst.IR = src.IR
		dst.fcr0 = src.fcr0
		dst.fcr25 = src.fcr25
		dst.fcr26 = src.fcr26
		dst.fcr27 = src.fcr27
		dst.fcr28 = src.fcr28
		dst.fcr31 = src.fcr31
		for (n in 0 until 32) dst.setGpr(n, src.getGpr(n))
		for (n in 0 until 32) dst.setFpr(n, src.getFpr(n))
		for (n in 0 until 128) dst.setVfpr(n, src.getVfpr(n))
	}

	val summary: String
		//get() = "REGS($id)[" + (0 until 32).map { "r%d=%d".format(it, getGpr(it)) }.joinToString(", ") + "]"
		get() = "REGS($id)[" + (0 until 32).map { "r%d=0x%08X".format(it, getGpr(it)) }.joinToString(", ") + "]"
}

/*
class CpuState(val mem: Memory, val syscalls: Syscalls = TraceSyscallHandler()) {
	var r0: Int; set(value) = Unit; get() = 0
	var r1: Int = 0
	var r2: Int = 0
	var r3: Int = 0
	var r4: Int = 0
	var r5: Int = 0
	var r6: Int = 0
	var r7: Int = 0
	var r8: Int = 0
	var r9: Int = 0
	var r10: Int = 0
	var r11: Int = 0
	var r12: Int = 0
	var r13: Int = 0
	var r14: Int = 0
	var r15: Int = 0
	var r16: Int = 0
	var r17: Int = 0
	var r18: Int = 0
	var r19: Int = 0
	var r20: Int = 0
	var r21: Int = 0
	var r22: Int = 0
	var r23: Int = 0
	var r24: Int = 0
	var r25: Int = 0
	var r26: Int = 0
	var r27: Int = 0
	var r28: Int = 0
	var r29: Int = 0
	var r30: Int = 0
	var r31: Int = 0

	val GPR = Gpr(this)

	var IR: Int = 0
	var _PC: Int = 0
	var _nPC: Int = 0
	var LO: Int = 0
	var HI: Int = 0
	var IC: Int = 0

	fun setPC(pc: Int) {
		_PC = pc
		_nPC = pc + 4
	}

	fun getPC() = _PC

	fun jump(pc: Int) {
		_PC = pc
		_nPC = pc + 4
	}

	fun advance_pc(offset: Int) {
		_PC = _nPC
		_nPC += offset
	}

	fun getGpr(index: Int): Int = GPR[index]
	fun setGpr(index: Int, v: Int): Unit {
		GPR[index] = value
	}

	class Gpr(val state: CpuState) {
		// ERROR!

		//fun ref(index: Int): KMutableProperty<Int> = state.run {
		//	when (index) {
		//		0 -> ::r0; 1 -> ::r1; 2 -> ::r2; 3 -> ::r3;
		//		4 -> ::r4; 5 -> ::r5; 6 -> ::r6; 7 -> ::r7;
		//		8 -> ::r8; 9 -> ::r9; 10 -> ::r10; 11 -> ::r11;
		//		12 -> ::r12; 13 -> ::r13; 14 -> ::r14; 15 -> ::r15;
		//		16 -> ::r16; 17 -> ::r17; 18 -> ::r18; 19 -> ::r19;
		//		20 -> ::r20; 21 -> ::r21; 22 -> ::r22; 23 -> ::r23;
		//		24 -> ::r24; 25 -> ::r25; 26 -> ::r26; 27 -> ::r27;
		//		28 -> ::r28; 29 -> ::r29; 30 -> ::r30; 31 -> ::r31
		//		else -> ::r0
		//	}
		//}

		fun hex(index: Int): String = "0x%08X".format(get(index))

		operator fun get(index: Int): Int = state.run {
			when (index and 0x1F) {
				0 -> r0; 1 -> r1; 2 -> r2; 3 -> r3
				4 -> r4; 5 -> r5; 6 -> r6; 7 -> r7
				8 -> r8; 9 -> r9; 10 -> r10; 11 -> r11
				12 -> r12; 13 -> r13; 14 -> r14; 15 -> r15
				16 -> r16; 17 -> r17; 18 -> r18; 19 -> r19
				20 -> r20; 21 -> r21; 22 -> r22; 23 -> r23
				24 -> r24; 25 -> r25; 26 -> r26; 27 -> r27
				28 -> r28; 29 -> r29; 30 -> r30; 31 -> r31
				else -> 0
			}
		}

		operator fun set(index: Int, v: Int): Unit = state.run {
			when (index and 0x1F) {
				0 -> r0 = v; 1 -> r1 = v; 2 -> r2 = v; 3 -> r3 = v
				4 -> r4 = v; 5 -> r5 = v; 6 -> r6 = v; 7 -> r7 = v
				8 -> r8 = v; 9 -> r9 = v; 10 -> r10 = v; 11 -> r11 = v
				12 -> r12 = v; 13 -> r13 = v; 14 -> r14 = v; 15 -> r15 = v
				16 -> r16 = v; 17 -> r17 = v; 18 -> r18 = v; 19 -> r19 = v
				20 -> r20 = v; 21 -> r21 = v; 22 -> r22 = v; 23 -> r23 = v
				24 -> r24 = v; 25 -> r25 = v; 26 -> r26 = v; 27 -> r27 = v
				28 -> r28 = v; 29 -> r29 = v; 30 -> r30 = v; 31 -> r31 = v
				else -> Unit
			}
		}
	}

	fun syscall(syscall: Int): Unit = syscalls.syscall(this, syscall)
}
*/
