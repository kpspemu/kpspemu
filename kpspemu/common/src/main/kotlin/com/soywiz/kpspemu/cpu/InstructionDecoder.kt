package com.soywiz.kpspemu.cpu

import com.soywiz.korio.util.extract
import com.soywiz.korio.util.signExtend

interface InstructionDecoder {
	val Int.lsb: Int get() = this.extract(6 + 5 * 0, 5)
	val Int.msb: Int get() = this.extract(6 + 5 * 1, 5)

	val Int.pos: Int get() = this.lsb
	val Int.rd: Int get() = this.extract(11 + 5 * 0, 5)
	val Int.rt: Int get() = this.extract(11 + 5 * 1, 5)
	val Int.rs: Int get() = this.extract(11 + 5 * 2, 5)
	val Int.syscall: Int get() = this.extract(6, 20)

	operator fun <T> CpuState.invoke(callback: CpuState.() -> T) = this.run(callback)

	var CpuState.RD: Int; get() = this.GPR[this.I.rd]; set(value) = run { this.GPR[this.I.rd] = value }
	var CpuState.RT: Int; get() = this.GPR[this.I.rt]; set(value) = run { this.GPR[this.I.rt] = value }
	var CpuState.RS: Int; get() = this.GPR[this.I.rs]; set(value) = run { this.GPR[this.I.rs] = value }

	//val CpuState.IMM16: Int; get() = I.extract()
	//val CpuState.U_IMM16: Int get() = TODO()

	val CpuState.S_IMM16: Int; get() = U_IMM16.signExtend(16)
	val CpuState.U_IMM16: Int get() = this.I.extract(0, 16)
	val CpuState.POS: Int get() = this.I.pos
	val CpuState.SYSCALL: Int get() = this.I.syscall
}

/*

export class Instruction {
	constructor(public PC: number, public data: number) {
	}

	static fromMemoryAndPC(memory: Memory, PC: number) { return new Instruction(PC, memory.readInt32(PC)); }

	extract(offset: number, length: number) { return BitUtils.extract(this.data, offset, length); }
	extract_s(offset: number, length: number) { return BitUtils.extractSigned(this.data, offset, length); }
	insert(offset: number, length: number, value: number) { this.data = BitUtils.insert(this.data, offset, length, value); }

	get rd() { return this.extract(11 + 5 * 0, 5); } set rd(value: number) { this.insert(11 + 5 * 0, 5, value); }
	get rt() { return this.extract(11 + 5 * 1, 5); } set rt(value: number) { this.insert(11 + 5 * 1, 5, value); }
	get rs() { return this.extract(11 + 5 * 2, 5); } set rs(value: number) { this.insert(11 + 5 * 2, 5, value); }

	get fd() { return this.extract(6 + 5 * 0, 5); } set fd(value: number) { this.insert(6 + 5 * 0, 5, value); }
	get fs() { return this.extract(6 + 5 * 1, 5); } set fs(value: number) { this.insert(6 + 5 * 1, 5, value); }
	get ft() { return this.extract(6 + 5 * 2, 5); } set ft(value: number) { this.insert(6 + 5 * 2, 5, value); }

	get VD() { return this.extract(0, 7); } set VD(value: number) { this.insert(0, 7, value); }
	get VS() { return this.extract(8, 7); } set VS(value: number) { this.insert(8, 7, value); }
	get VT() { return this.extract(16, 7); } set VT(value: number) { this.insert(16, 7, value); }
	get VT5_1() { return this.VT5 | (this.VT1 << 5); } set VT5_1(value: number) { this.VT5 = value; this.VT1 = (value >>> 5); }
	get IMM14() { return this.extract_s(2, 14); } set IMM14(value: number) { this.insert(2, 14, value); }

	get ONE() { return this.extract(7, 1); } set ONE(value: number) { this.insert(7, 1, value); }
	get TWO() { return this.extract(15, 1); } set TWO(value: number) { this.insert(15, 1, value); }
	get ONE_TWO() { return (1 + 1 * this.ONE + 2 * this.TWO); } set ONE_TWO(value: number) { this.ONE = (((value - 1) >>> 0) & 1); this.TWO = (((value - 1) >>> 1) & 1); }


	get IMM8() { return this.extract(16, 8); } set IMM8(value: number) { this.insert(16, 8, value); }
	get IMM5() { return this.extract(16, 5); } set IMM5(value: number) { this.insert(16, 5, value); }
	get IMM3() { return this.extract(18, 3); } set IMM3(value: number) { this.insert(18, 3, value); }
	get IMM7() { return this.extract(0, 7); } set IMM7(value: number) { this.insert(0, 7, value); }
	get IMM4() { return this.extract(0, 4); } set IMM4(value: number) { this.insert(0, 4, value); }
	get VT1() { return this.extract(0, 1); } set VT1(value: number) { this.insert(0, 1, value); }
	get VT2() { return this.extract(0, 2); } set VT2(value: number) { this.insert(0, 2, value); }
	get VT5() { return this.extract(16, 5); } set VT5(value: number) { this.insert(16, 5, value); }
	get VT5_2() { return this.VT5 | (this.VT2 << 5); }
	get IMM_HF() { return HalfFloat.toFloat(this.imm16); }

	get pos() { return this.lsb; } set pos(value: number) { this.lsb = value; }
	get size_e() { return this.msb + 1; } set size_e(value: number) { this.msb = value - 1; }
	get size_i() { return this.msb - this.lsb + 1; } set size_i(value: number) { this.msb = this.lsb + value - 1; }

	get lsb() { return this.extract(6 + 5 * 0, 5); } set lsb(value: number) { this.insert(6 + 5 * 0, 5, value); }
	get msb() { return this.extract(6 + 5 * 1, 5); } set msb(value: number) { this.insert(6 + 5 * 1, 5, value); }
	get c1cr() { return this.extract(6 + 5 * 1, 5); } set c1cr(value: number) { this.insert(6 + 5 * 1, 5, value); }

	get syscall() { return this.extract(6, 20); } set syscall(value: number) { this.insert(6, 20, value); }

	get imm16() { var res = this.u_imm16; if (res & 0x8000) res |= 0xFFFF0000; return res; } set imm16(value: number) { this.insert(0, 16, value); }
	get u_imm16() { return this.extract(0, 16); } set u_imm16(value: number) { this.insert(0, 16, value); }
	get u_imm26() { return this.extract(0, 26); } set u_imm26(value: number) { this.insert(0, 26, value); }

	get jump_bits() { return this.extract(0, 26); } set jump_bits(value: number) { this.insert(0, 26, value); }
	get jump_real() { return (this.jump_bits * 4) >>> 0; } set jump_real(value: number) { this.jump_bits = (value / 4) >>> 0; }

	set branch_address(value:number) { this.imm16 = (value - this.PC - 4) / 4; }
	set jump_address(value:number) { this.u_imm26 = value / 4; }

	get branch_address() { return this.PC + this.imm16 * 4 + 4; }
	get jump_address() { return this.u_imm26 * 4; }
}
 */