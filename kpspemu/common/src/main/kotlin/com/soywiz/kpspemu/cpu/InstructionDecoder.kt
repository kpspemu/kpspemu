package com.soywiz.kpspemu.cpu

import com.soywiz.kmem.*

data class InstructionData(var value: Int, var pc: Int) {
    var lsb: Int get() = value.extract(6, 5); set(v) = run { value = value.insert(v, 6, 5) }
    var msb: Int get() = value.extract(11, 5); set(v) = run { value = value.insert(v, 11, 5) }
    var pos: Int get() = value.extract(6, 5); set(v) = run { value = value.insert(v, 6, 5) }

    var s_imm16: Int get() = value.extract(0, 16) shl 16 shr 16; set(v) = run { value = value.insert(v, 0, 16) }
    var u_imm16: Int get() = value.extract(0, 16); set(v) = run { value = value.insert(v, 0, 16) }
    var u_imm26: Int get() = value.extract(0, 26); set(v) = run { value = value.insert(v, 0, 26) }

    var rd: Int get() = value.extract(11, 5); set(v) = run { value = value.insert(v, 11, 5) }
    var rt: Int get() = value.extract(16, 5); set(v) = run { value = value.insert(v, 16, 5) }
    var rs: Int get() = value.extract(21, 5); set(v) = run { value = value.insert(v, 21, 5) }

    var jump_address get() = u_imm26 * 4; set(v) = run { u_imm26 = v / 4 }
}

// https://www.cs.umd.edu/users/meesh/411/SimpleMips.htm
// http://www.mrc.uidaho.edu/mrc/people/jff/digital/MIPSir.html
// https://electronics.stackexchange.com/questions/28444/mips-pic32-branch-vs-branch-likely
open class InstructionDecoder {
    //val Int.lsb: Int get() = this.extract(6 + 5 * 0, 5)
    //val Int.msb: Int get() = this.extract(6 + 5 * 1, 5)
    //val Int.pos: Int get() = this.lsb

    //val Int.rd: Int get() = this.extract(11 + 5 * 0, 5)
    //val Int.rt: Int get() = this.extract(11 + 5 * 1, 5)
    //val Int.rs: Int get() = this.extract(11 + 5 * 2, 5)

    inline val Int.lsb: Int get() = (this ushr 6) and 0x1F
    inline val Int.msb: Int get() = (this ushr 11) and 0x1F
    inline val Int.pos: Int get() = lsb

    inline val Int.size_e: Int get() = msb + 1
    inline val Int.size_i: Int get() = msb - lsb + 1

    inline val Int.rd: Int get() = (this ushr 11) and 0x1F
    inline val Int.rt: Int get() = (this ushr 16) and 0x1F
    inline val Int.rs: Int get() = (this ushr 21) and 0x1F

    inline val Int.fd: Int get() = (this ushr 6) and 0x1F
    inline val Int.fs: Int get() = (this ushr 11) and 0x1F
    inline val Int.ft: Int get() = (this ushr 16) and 0x1F

    inline val Int.vd: Int get() = (this ushr 0) and 0x7F
    inline val Int.vs: Int get() = (this ushr 8) and 0x7F
    inline val Int.vt: Int get() = (this ushr 16) and 0x7F
    inline val Int.vt1: Int get() = this.extract(0, 1)
    inline val Int.vt2: Int get() = this.extract(0, 2)
    inline val Int.vt5: Int get() = this.extract(16, 5)
    inline val Int.vt5_1: Int get() = vt5 or (vt1 shl 5)
    inline val Int.vt5_2: Int get() = vt5 or (vt2 shl 5)

    val Int.imm8: Int get() = this.extract(16, 8)
    val Int.imm5: Int get() = this.extract(16, 5)
    val Int.imm3: Int get() = this.extract(16, 3)
    val Int.imm7: Int get() = this.extract(0, 7)
    val Int.imm4: Int get() = this.extract(0, 4)

    inline val Int.one: Int get() = this.extract(7, 1)
    inline val Int.two: Int get() = this.extract(15, 1)
    inline val Int.one_two: Int get() = (1 + 1 * this.one + 2 * this.two)

    inline val Int.syscall: Int get() = this.extract(6, 20)
    inline val Int.s_imm16: Int get() = (this shl 16) shr 16
    inline val Int.u_imm16: Int get() = this and 0xFFFF

    inline val Int.s_imm14: Int get() = this.extract(2, 14).signExtend(14)

    inline val Int.u_imm26: Int get() = this.extract(0, 26)
    inline val Int.jump_address: Int get() = u_imm26 * 4

    inline operator fun CpuState.invoke(callback: CpuState.() -> Unit) = this.normal(callback)

    inline fun CpuState.preadvance(callback: CpuState.() -> Unit) {
        advance_pc(4)
        this.run(callback)
    }

    inline fun CpuState.normal(callback: CpuState.() -> Unit) {
        this.run(callback)
        advance_pc(4)
    }

    inline fun CpuState.none(callback: CpuState.() -> Unit) {
        this.run(callback)
    }

    inline fun CpuState.branch(callback: CpuState.() -> Boolean) {
        val result = this.run(callback)

        // beq
        // if $s == $t advance_pc (offset << 2)); else advance_pc (4);
        if (result) {
            advance_pc(S_IMM16 * 4)
        } else {
            advance_pc(4)
        }
    }

    inline fun CpuState.branchLikely(callback: CpuState.() -> Boolean) {
        val result = this.run(callback)
        if (result) {
            //println("YAY!: $S_IMM16")
            advance_pc(S_IMM16 * 4)
        } else {
            //println("NON!: $S_IMM16")
            //println("-- %08X".format(_nPC))
            //println("-- %08X".format(_PC))
            _PC = _nPC + 4
            _nPC = _PC + 4
        }
    }

    inline var CpuState.VT: Float; get() = getVfpr(IR.vt); set(value) = run { setVfpr(IR.vt, value) }
    inline var CpuState.VD: Float; get() = getVfpr(IR.vd); set(value) = run { setVfpr(IR.vd, value) }
    inline var CpuState.VS: Float; get() = getVfpr(IR.vs); set(value) = run { setVfpr(IR.vs, value) }

    inline var CpuState.VT_I: Int; get() = getVfprI(IR.vt); set(value) = run { setVfprI(IR.vt, value) }
    inline var CpuState.VD_I: Int; get() = getVfprI(IR.vd); set(value) = run { setVfprI(IR.vd, value) }
    inline var CpuState.VS_I: Int; get() = getVfprI(IR.vs); set(value) = run { setVfprI(IR.vs, value) }

    inline var CpuState.RD: Int; get() = getGpr(IR.rd); set(value) = run { setGpr(IR.rd, value) }
    inline var CpuState.RT: Int; get() = getGpr(IR.rt); set(value) = run { setGpr(IR.rt, value) }
    inline var CpuState.RS: Int; get() = getGpr(IR.rs); set(value) = run { setGpr(IR.rs, value) }

    inline var CpuState.FD: Float; get() = getFpr(IR.fd); set(value) = run { setFpr(IR.fd, value) }
    inline var CpuState.FT: Float; get() = getFpr(IR.ft); set(value) = run { setFpr(IR.ft, value) }
    inline var CpuState.FS: Float; get() = getFpr(IR.fs); set(value) = run { setFpr(IR.fs, value) }

    inline var CpuState.FD_I: Int; get() = getFprI(IR.fd); set(value) = run { setFprI(IR.fd, value) }
    inline var CpuState.FT_I: Int; get() = getFprI(IR.ft); set(value) = run { setFprI(IR.ft, value) }
    inline var CpuState.FS_I: Int; get() = getFprI(IR.fs); set(value) = run { setFprI(IR.fs, value) }

    inline val CpuState.RS_IMM16: Int; get() = RS + S_IMM16

    inline val CpuState.RS_IMM14: Int; get() = RS + S_IMM14 * 4

    //val CpuState.IMM16: Int; get() = I.extract()
    //val CpuState.U_IMM16: Int get() = TODO()

    inline val CpuState.S_IMM14: Int; get() = IR.s_imm14
    inline val CpuState.S_IMM16: Int; get() = IR.s_imm16
    inline val CpuState.U_IMM16: Int get() = IR.u_imm16
    inline val CpuState.POS: Int get() = IR.pos
    inline val CpuState.SIZE_E: Int get() = IR.size_e
    inline val CpuState.SIZE_I: Int get() = IR.size_i
    inline val CpuState.SYSCALL: Int get() = IR.syscall
    inline val CpuState.U_IMM26: Int get() = IR.u_imm26
    inline val CpuState.JUMP_ADDRESS: Int get() = IR.jump_address
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