package com.soywiz.kpspemu.cpu

import com.soywiz.kpspemu.mem.Memory

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

	fun setPC(pc: Int) = jump(pc)
	fun getPC() = _PC

	fun jump(pc: Int) {
		_PC = pc
		_nPC = pc + 4
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

		operator fun get(index: Int): Int = state.run {
			when (index) {
				0 -> r0; 1 -> r1; 2 -> r2; 3 -> r3;
				4 -> r4; 5 -> r5; 6 -> r6; 7 -> r7;
				8 -> r8; 9 -> r9; 10 -> r10; 11 -> r11;
				12 -> r12; 13 -> r13; 14 -> r14; 15 -> r15;
				16 -> r16; 17 -> r17; 18 -> r18; 19 -> r19;
				20 -> r20; 21 -> r21; 22 -> r22; 23 -> r23;
				24 -> r24; 25 -> r25; 26 -> r26; 27 -> r27;
				28 -> r28; 29 -> r29; 30 -> r30; 31 -> r31
				else -> 0
			}
		}

		operator fun set(index: Int, v: Int): Unit = state.run {
			when (index) {
				0 -> r0 = v; 1 -> r1 = v; 2 -> r2 = v; 3 -> r3 = v;
				4 -> r4 = v; 5 -> r5 = v; 6 -> r6 = v; 7 -> r7 = v;
				8 -> r8 = v; 9 -> r9 = v; 10 -> r10 = v; 11 -> r11 = v;
				12 -> r12 = v; 13 -> r13 = v; 14 -> r14 = v; 15 -> r15 = v;
				16 -> r16 = v; 17 -> r17 = v; 18 -> r18 = v; 19 -> r19 = v;
				20 -> r20 = v; 21 -> r21 = v; 22 -> r22 = v; 23 -> r23 = v;
				24 -> r24 = v; 25 -> r25 = v; 26 -> r26 = v; 27 -> r27 = v;
				28 -> r28 = v; 29 -> r29 = v; 30 -> r30 = v; 31 -> r31 = v
				else -> Unit
			}
		}
	}

	fun syscall(syscall: Int): Unit = syscalls.syscall(this, syscall)
}