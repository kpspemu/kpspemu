package com.soywiz.kpspemu.cpu

import com.soywiz.korio.util.Extra
import com.soywiz.korio.util.extract
import com.soywiz.korio.util.insert
import com.soywiz.kpspemu.mem.Memory

data class CpuBreakException(val id: Int) : Exception() {
	companion object {
		val THREAD_WAIT = 10001
		val THREAD_EXIT_KILL = 10002
	}
}

// http://www.cs.uwm.edu/classes/cs315/Bacon/Lecture/HTML/ch05s03.html
var CpuState.K0: Int; set(value) = run { r26 = value }; get() = r26
var CpuState.K1: Int; set(value) = run { r27 = value }; get() = r27
var CpuState.GP: Int; set(value) = run { r28 = value }; get() = r28
var CpuState.SP: Int; set(value) = run { r29 = value }; get() = r29
var CpuState.FP: Int; set(value) = run { r30 = value }; get() = r30
var CpuState.RA: Int; set(value) = run { r31 = value }; get() = r31

class CpuState(val mem: Memory, val syscalls: Syscalls = TraceSyscallHandler()) : Extra by Extra.Mixin() {
	var _R = IntArray(32)
	var _F = com.soywiz.korio.mem.FastMemory.alloc(32 * 4)
	val _VFPR = com.soywiz.korio.mem.FastMemory.alloc(128 * 4)

	var fcr0: Int = 0x00003351
	var fcr25: Int = 0
	var fcr26: Int = 0
	var fcr27: Int = 0
	var fcr28: Int = 0
	var fcr31: Int = 0x00000e00

	fun updateFCR31(value: Int) {
		fcr31 = value and 0x0183FFFF
	}

	var fcr31_rm: Int set(value) = run { fcr31 = fcr31.insert(value, 0, 2) }; get() = fcr31.extract(0, 2)
	var fcr31_2_21: Int set(value) = run { fcr31 = fcr31.insert(value, 2, 21) }; get() = fcr31.extract(2, 21)
	var fcr31_cc: Boolean set(value) = run { fcr31 = fcr31.insert(value, 23) }; get() = fcr31.extract(23)
	var fcr31_fs: Boolean set(value) = run { fcr31 = fcr31.insert(value, 24) }; get() = fcr31.extract(24)
	var fcr31_25_7: Int set(value) = run { fcr31 = fcr31.insert(value, 25, 7) }; get() = fcr31.extract(25, 7)

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

	val GPR = Gpr(this)
	//val FPR = FloatArray(32) { 0f }
	//val FPR_I = FprI(this)

	var IR: Int = 0
	var _PC: Int = 0
	var _nPC: Int = 0
	var LO: Int = 0
	var HI: Int = 0
	var IC: Int = 0

	val PC: Int get() = _PC

	var HI_LO: Long
		get() = (HI.toLong() shl 32) or (LO.toLong() and 0xFFFFFFFF)
		set(value) {
			HI = (value ushr 32).toInt()
			LO = (value ushr 0).toInt()
		}

	fun setPC(pc: Int) {
		_PC = pc
		_nPC = pc + 4
	}

	fun jump(pc: Int) {
		_PC = pc
		_nPC = pc + 4
	}

	fun advance_pc(offset: Int) {
		_PC = _nPC
		_nPC += offset
	}

	fun getGpr(index: Int): Int = _R[index and 0x1F]
	fun setGpr(index: Int, v: Int): Unit {
		if (index != 0) {
			_R[index and 0x1F] = v
		}
	}

	fun getFpr(index: Int): Float = _F.getAlignedFloat32(index)
	fun setFpr(index: Int, v: Float): Unit = run { _F.setAlignedFloat32(index, v) }
	fun getFprI(index: Int): Int = _F.getAlignedInt32(index)
	fun setFprI(index: Int, v: Int): Unit = run { _F.setAlignedInt32(index, v) }

	fun setVfpr(index: Int, value: Float) = _VFPR.setAlignedFloat32(index, value)
	fun getVfpr(index: Int): Float = _VFPR.getAlignedFloat32(index)

	fun setVfprI(index: Int, value: Int) = _VFPR.setAlignedInt32(index, value)
	fun getVfprI(index: Int): Int = _VFPR.getAlignedInt32(index)

	class Gpr(val state: CpuState) {
		operator fun get(index: Int): Int = state.getGpr(index)
		operator fun set(index: Int, v: Int): Unit = state.setGpr(index, v)
	}

	class FprI(val state: CpuState) {
		operator fun get(index: Int): Int = state.getFprI(index)
		operator fun set(index: Int, v: Int): Unit = state.setFprI(index, v)
	}

	fun syscall(syscall: Int): Unit = syscalls.syscall(this, syscall)
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
