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

    val Int.lsb: Int get() = (this ushr 6) and 0x1F
    val Int.msb: Int get() = (this ushr 11) and 0x1F
    val Int.pos: Int get() = lsb

    val Int.size_e: Int get() = msb + 1
    val Int.size_i: Int get() = msb - lsb + 1

    val Int.rd: Int get() = (this ushr 11) and 0x1F
    val Int.rt: Int get() = (this ushr 16) and 0x1F
    val Int.rs: Int get() = (this ushr 21) and 0x1F

    val Int.fd: Int get() = (this ushr 6) and 0x1F
    val Int.fs: Int get() = (this ushr 11) and 0x1F
    val Int.ft: Int get() = (this ushr 16) and 0x1F

    val Int.vd: Int get() = (this ushr 0) and 0x7F
    val Int.vs: Int get() = (this ushr 8) and 0x7F
    val Int.vt: Int get() = (this ushr 16) and 0x7F
    val Int.vt1: Int get() = this.extract(0, 1)
    val Int.vt2: Int get() = this.extract(0, 2)
    val Int.vt5: Int get() = this.extract(16, 5)
    val Int.vt5_1: Int get() = vt5 or (vt1 shl 5)
    val Int.vt5_2: Int get() = vt5 or (vt2 shl 5)

    val Int.imm8: Int get() = this.extract(16, 8)
    val Int.imm5: Int get() = this.extract(16, 5)
    val Int.imm3: Int get() = this.extract(16, 3)
    val Int.imm7: Int get() = this.extract(0, 7)
    val Int.imm4: Int get() = this.extract(0, 4)

    val Int.one: Int get() = this.extract(7, 1)
    val Int.two: Int get() = this.extract(15, 1)
    val Int.one_two: Int get() = (1 + 1 * this.one + 2 * this.two)

    val Int.syscall: Int get() = this.extract(6, 20)
    val Int.s_imm16: Int get() = (this shl 16) shr 16
    val Int.u_imm16: Int get() = this and 0xFFFF

    val Int.s_imm14: Int get() = this.extract(2, 14).signExtend(14)

    val Int.u_imm26: Int get() = this.extract(0, 26)
    val Int.jump_address: Int get() = u_imm26 * 4

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

    var CpuState.VT: Float; get() = getVfpr(IR.vt); set(value) = run { setVfpr(IR.vt, value) }
    var CpuState.VD: Float; get() = getVfpr(IR.vd); set(value) = run { setVfpr(IR.vd, value) }
    var CpuState.VS: Float; get() = getVfpr(IR.vs); set(value) = run { setVfpr(IR.vs, value) }

    var CpuState.VT_I: Int; get() = getVfprI(IR.vt); set(value) = run { setVfprI(IR.vt, value) }
    var CpuState.VD_I: Int; get() = getVfprI(IR.vd); set(value) = run { setVfprI(IR.vd, value) }
    var CpuState.VS_I: Int; get() = getVfprI(IR.vs); set(value) = run { setVfprI(IR.vs, value) }

    var CpuState.RD: Int; get() = regs.getGpr(IR.rd); set(value) = run { regs.setGpr(IR.rd, value) }
    var CpuState.RT: Int; get() = regs.getGpr(IR.rt); set(value) = run { regs.setGpr(IR.rt, value) }
    var CpuState.RS: Int; get() = regs.getGpr(IR.rs); set(value) = run { regs.setGpr(IR.rs, value) }

    var CpuState.FD: Float; get() = getFpr(IR.fd); set(value) = run { setFpr(IR.fd, value) }
    var CpuState.FT: Float; get() = getFpr(IR.ft); set(value) = run { setFpr(IR.ft, value) }
    var CpuState.FS: Float; get() = getFpr(IR.fs); set(value) = run { setFpr(IR.fs, value) }

    var CpuState.FD_I: Int; get() = getFprI(IR.fd); set(value) = run { setFprI(IR.fd, value) }
    var CpuState.FT_I: Int; get() = getFprI(IR.ft); set(value) = run { setFprI(IR.ft, value) }
    var CpuState.FS_I: Int; get() = getFprI(IR.fs); set(value) = run { setFprI(IR.fs, value) }

    val CpuState.RS_IMM16: Int; get() = RS + S_IMM16

    val CpuState.RS_IMM14: Int; get() = RS + S_IMM14 * 4

    //val CpuState.IMM16: Int; get() = I.extract()
    //val CpuState.U_IMM16: Int get() = TODO()

    val CpuState.S_IMM14: Int; get() = IR.s_imm14
    val CpuState.S_IMM16: Int; get() = IR.s_imm16
    val CpuState.U_IMM16: Int get() = IR.u_imm16
    val CpuState.POS: Int get() = IR.pos
    val CpuState.SIZE_E: Int get() = IR.size_e
    val CpuState.SIZE_I: Int get() = IR.size_i
    val CpuState.SYSCALL: Int get() = IR.syscall
    val CpuState.U_IMM26: Int get() = IR.u_imm26
    val CpuState.JUMP_ADDRESS: Int get() = IR.jump_address
}
