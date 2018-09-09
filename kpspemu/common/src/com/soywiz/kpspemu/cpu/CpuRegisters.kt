package com.soywiz.kpspemu.cpu

import com.soywiz.dynarek2.*
import com.soywiz.kmem.*

inline class CpuRegisters(val data: D2MemoryInt)

// @TODO: Free
fun CpuRegisters(): CpuRegisters = CpuRegisters(D2MemoryInt(NewD2Memory(512 * 4).mem)).apply { init() }

// 0-32
inline var CpuRegisters.IR: InstructionRegister
    get() = InstructionRegister(data[0]);
    set(value) {
        data[0] = value.data
    }
inline var CpuRegisters.PC: Int get() = data[1]; set(value) = run { data[1] = value }
inline var CpuRegisters.nPC: Int get() = data[2]; set(value) = run { data[2] = value }
inline var CpuRegisters.LO: Int get() = data[3]; set(value) = run { data[3] = value }
inline var CpuRegisters.HI: Int get() = data[4]; set(value) = run { data[4] = value }
inline var CpuRegisters.IC: Int get() = data[5]; set(value) = run { data[5] = value }

// 32-64
inline fun CpuRegisters.setGpr(index: Int, value: Int) {
    if (index != 0) {
        data[32 + index] = value
    }
}

inline fun CpuRegisters.getGpr(index: Int) = data[32 + index]

// 64-96
inline fun CpuRegisters.setFpr(index: Int, value: Float) = run { if (index != 0) data.setFloat(64 + index, value) }

inline fun CpuRegisters.getFpr(index: Int): Float = data.getFloat(64 + index)

inline fun CpuRegisters.setFprI(index: Int, value: Int) = run { if (index != 0) data[64 + index] = value }
inline fun CpuRegisters.getFprI(index: Int): Int = data[64 + index]

// 96-112
inline fun CpuRegisters.getVfprC(index: Int): Int = data[96 + index]

inline fun CpuRegisters.setVfprC(index: Int, value: Int) = run { data[96 + index] = value }

inline fun CpuRegisters.init() {
    setVfprC(0, 0)
    setVfprC(1, 0)
    setVfprC(2, 0)
    setVfprC(3, 0xFF)
    setVfprC(4, 0)
    setVfprC(5, 0)
    setVfprC(6, 0)
    setVfprC(7, 0)
    setVfprC(8, 0x3F800000)
    setVfprC(9, 0x3F800000)
    setVfprC(10, 0x3F800000)
    setVfprC(11, 0x3F800000)
    setVfprC(12, 0x3F800000)
    setVfprC(13, 0x3F800000)
    setVfprC(14, 0x3F800000)
    setVfprC(15, 0x3F800000)

    setVfprCC(0, 0x00003351)
    setVfprCC(25, 0)
    setVfprCC(26, 0)
    setVfprCC(27, 0)
    setVfprCC(28, 0)
    setVfprCC(31, 0x00000e00)
}

// 128-256
inline fun CpuRegisters.getVfpr(index: Int): Float = data.getFloat(128 + index)

inline fun CpuRegisters.setVfpr(index: Int, value: Float) = data.setFloat(128 + index, value)

// 256-272
inline fun CpuRegisters.getVfprCC(index: Int): Int = data[256 + index]

inline fun CpuRegisters.setVfprCC(index: Int, value: Int) = run { data[256 + index] = value }


inline var CpuRegisters.VFPR_CC: Int
    get() = getVfprC(CpuState.VFPU_CTRL.CC);
    set(value) = run {
        setVfprC(
            CpuState.VFPU_CTRL.CC,
            value
        )
    }

inline fun CpuRegisters.VFPR_CC(index: Int) = VFPR_CC.extract(index)

inline var CpuRegisters.fcr0: Int get() = getVfprCC(0); set(value) = run { setVfprCC(0, value) }
inline var CpuRegisters.fcr25: Int get() = getVfprCC(25); set(value) = run { setVfprCC(25, value) }
inline var CpuRegisters.fcr26: Int get() = getVfprCC(26); set(value) = run { setVfprCC(26, value) }
inline var CpuRegisters.fcr27: Int get() = getVfprCC(27); set(value) = run { setVfprCC(27, value) }
inline var CpuRegisters.fcr28: Int get() = getVfprCC(28); set(value) = run { setVfprCC(28, value) }
inline var CpuRegisters.fcr31: Int get() = getVfprCC(31); set(value) = run { setVfprCC(31, value) }

//var vpfxs = VfpuSourceTargetPrefix(VFPRC, CpuState.VFPU_CTRL.SPREFIX)
//var vpfxt = VfpuSourceTargetPrefix(VFPRC, CpuState.VFPU_CTRL.TPREFIX)
//var vpfxd = VfpuDestinationPrefix(VFPRC, CpuState.VFPU_CTRL.DPREFIX)

inline fun CpuRegisters.updateFCR31(value: Int) {
    fcr31 = value and 0x0183FFFF
}

inline var CpuRegisters.fcr31_rm: Int
    set(value) = run { fcr31 = fcr31.insert(value, 0, 2) };
    get() = fcr31.extract(
        0,
        2
    )
inline var CpuRegisters.fcr31_2_21: Int
    set(value) = run {
        fcr31 = fcr31.insert(value, 2, 21)
    };
    get() = fcr31.extract(2, 21)
inline var CpuRegisters.fcr31_cc: Boolean
    set(value) = run {
        fcr31 = fcr31.insert(value, 23)
    };
    get() = fcr31.extract(23)
inline var CpuRegisters.fcr31_fs: Boolean
    set(value) = run {
        fcr31 = fcr31.insert(value, 24)
    };
    get() = fcr31.extract(24)
inline var CpuRegisters.fcr31_25_7: Int
    set(value) = run { fcr31 = fcr31.insert(value, 25, 7) };
    get() = fcr31.extract(
        25,
        7
    )

inline var CpuRegisters.RS: Int get() = getGpr(IR.rs); set(value) = run { setGpr(IR.rs, value) }
inline var CpuRegisters.RD: Int get() = getGpr(IR.rd); set(value) = run { setGpr(IR.rd, value) }
inline var CpuRegisters.RT: Int get() = getGpr(IR.rt); set(value) = run { setGpr(IR.rt, value) }

inline var CpuRegisters.FS: Float get() = getFpr(IR.fs); set(value) = run { setFpr(IR.fs, value) }
inline var CpuRegisters.FD: Float get() = getFpr(IR.fd); set(value) = run { setFpr(IR.fd, value) }
inline var CpuRegisters.FT: Float get() = getFpr(IR.ft); set(value) = run { setFpr(IR.ft, value) }

inline var CpuRegisters.FS_I: Int get() = getFprI(IR.fs); set(value) = run { setFprI(IR.fs, value) }
inline var CpuRegisters.FD_I: Int get() = getFprI(IR.fd); set(value) = run { setFprI(IR.fd, value) }
inline var CpuRegisters.FT_I: Int get() = getFprI(IR.ft); set(value) = run { setFprI(IR.ft, value) }

inline val CpuRegisters.RS_IMM16: Int get() = RS + IR.s_imm16
inline val CpuRegisters.RS_IMM14: Int get() = RS + IR.s_imm14

inline val CpuRegisters.POS: Int get() = IR.pos
inline val CpuRegisters.SIZE_E: Int get() = IR.size_e
inline val CpuRegisters.SIZE_I: Int get() = IR.size_i
inline val CpuRegisters.U_IMM16: Int get() = IR.u_imm16
inline val CpuRegisters.S_IMM16: Int get() = IR.s_imm16
inline val CpuRegisters.SYSCALL: Int get() = IR.syscall
inline val CpuRegisters.JUMP_ADDRESS: Int get() = IR.jump_address

inline var CpuRegisters.r0: Int; set(value) = run { setGpr(0, value) }; get() = getGpr(0)
inline var CpuRegisters.r1: Int; set(value) = run { setGpr(1, value) }; get() = getGpr(1)
inline var CpuRegisters.r2: Int; set(value) = run { setGpr(2, value) }; get() = getGpr(2)
inline var CpuRegisters.r3: Int; set(value) = run { setGpr(3, value) }; get() = getGpr(3)
inline var CpuRegisters.r4: Int; set(value) = run { setGpr(4, value) }; get() = getGpr(4)
inline var CpuRegisters.r5: Int; set(value) = run { setGpr(5, value) }; get() = getGpr(5)
inline var CpuRegisters.r6: Int; set(value) = run { setGpr(6, value) }; get() = getGpr(6)
inline var CpuRegisters.r7: Int; set(value) = run { setGpr(7, value) }; get() = getGpr(7)
inline var CpuRegisters.r8: Int; set(value) = run { setGpr(8, value) }; get() = getGpr(8)
inline var CpuRegisters.r9: Int; set(value) = run { setGpr(9, value) }; get() = getGpr(9)
inline var CpuRegisters.r10: Int; set(value) = run { setGpr(10, value) }; get() = getGpr(10)
inline var CpuRegisters.r11: Int; set(value) = run { setGpr(11, value) }; get() = getGpr(11)
inline var CpuRegisters.r12: Int; set(value) = run { setGpr(12, value) }; get() = getGpr(12)
inline var CpuRegisters.r13: Int; set(value) = run { setGpr(13, value) }; get() = getGpr(13)
inline var CpuRegisters.r14: Int; set(value) = run { setGpr(14, value) }; get() = getGpr(14)
inline var CpuRegisters.r15: Int; set(value) = run { setGpr(15, value) }; get() = getGpr(15)
inline var CpuRegisters.r16: Int; set(value) = run { setGpr(16, value) }; get() = getGpr(16)
inline var CpuRegisters.r17: Int; set(value) = run { setGpr(17, value) }; get() = getGpr(17)
inline var CpuRegisters.r18: Int; set(value) = run { setGpr(18, value) }; get() = getGpr(18)
inline var CpuRegisters.r19: Int; set(value) = run { setGpr(19, value) }; get() = getGpr(19)
inline var CpuRegisters.r20: Int; set(value) = run { setGpr(20, value) }; get() = getGpr(20)
inline var CpuRegisters.r21: Int; set(value) = run { setGpr(21, value) }; get() = getGpr(21)
inline var CpuRegisters.r22: Int; set(value) = run { setGpr(22, value) }; get() = getGpr(22)
inline var CpuRegisters.r23: Int; set(value) = run { setGpr(23, value) }; get() = getGpr(23)
inline var CpuRegisters.r24: Int; set(value) = run { setGpr(24, value) }; get() = getGpr(24)
inline var CpuRegisters.r25: Int; set(value) = run { setGpr(25, value) }; get() = getGpr(25)
inline var CpuRegisters.r26: Int; set(value) = run { setGpr(26, value) }; get() = getGpr(26)
inline var CpuRegisters.r27: Int; set(value) = run { setGpr(27, value) }; get() = getGpr(27)
inline var CpuRegisters.r28: Int; set(value) = run { setGpr(28, value) }; get() = getGpr(28)
inline var CpuRegisters.r29: Int; set(value) = run { setGpr(29, value) }; get() = getGpr(29)
inline var CpuRegisters.r30: Int; set(value) = run { setGpr(30, value) }; get() = getGpr(30)
inline var CpuRegisters.r31: Int; set(value) = run { setGpr(31, value) }; get() = getGpr(31)
inline var CpuRegisters.r32: Int; set(value) = run { setGpr(32, value) }; get() = getGpr(32)

inline var CpuRegisters.K0: Int; set(value) = run { r26 = value }; get() = r26
inline var CpuRegisters.K1: Int; set(value) = run { r27 = value }; get() = r27
inline var CpuRegisters.GP: Int; set(value) = run { r28 = value }; get() = r28
inline var CpuRegisters.SP: Int; set(value) = run { r29 = value }; get() = r29
inline var CpuRegisters.FP: Int; set(value) = run { r30 = value }; get() = r30
inline var CpuRegisters.RA: Int; set(value) = run { r31 = value }; get() = r31

inline var CpuRegisters.V0: Int; set(value) = run { r2 = value }; get() = r2
inline var CpuRegisters.V1: Int; set(value) = run { r3 = value }; get() = r3
inline var CpuRegisters.A0: Int; set(value) = run { r4 = value }; get() = r4
inline var CpuRegisters.A1: Int; set(value) = run { r5 = value }; get() = r5
inline var CpuRegisters.A2: Int; set(value) = run { r6 = value }; get() = r6
inline var CpuRegisters.A3: Int; set(value) = run { r7 = value }; get() = r7

inline fun CpuRegisters.advance_pc(offset: Int) {
    PC = nPC
    nPC += offset
}

inline operator fun CpuRegisters.invoke(callback: CpuRegisters.() -> Unit) = normal(callback)

inline fun CpuRegisters.preadvance(callback: CpuRegisters.() -> Unit) {
    advance_pc(4)
    callback()
}

inline fun CpuRegisters.normal(callback: CpuRegisters.() -> Unit) {
    callback()
    advance_pc(4)
}

inline fun CpuRegisters.none(callback: CpuRegisters.() -> Unit) {
    callback()
}

inline fun CpuRegisters.branch(callback: CpuRegisters.() -> Boolean) {
    val result = callback()

    // beq
    // if $s == $t advance_pc (offset << 2)); else advance_pc (4);
    if (result) {
        advance_pc(S_IMM16 * 4)
    } else {
        advance_pc(4)
    }
}

inline fun CpuRegisters.branchLikely(callback: CpuRegisters.() -> Boolean) {
    val result = callback()
    if (result) {
        //println("YAY!: $S_IMM16")
        advance_pc(S_IMM16 * 4)
    } else {
        //println("NON!: $S_IMM16")
        //println("-- %08X".format(_nPC))
        //println("-- %08X".format(_PC))
        PC = nPC + 4
        nPC = PC + 4
    }
}
