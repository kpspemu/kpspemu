package com.soywiz.kpspemu.cpu

import com.soywiz.kmem.*

inline class InstructionRegister(val data: Int)

inline val InstructionRegister.lsb: Int get() = (data ushr 6) and 0x1F
inline val InstructionRegister.msb: Int get() = (data ushr 11) and 0x1F
inline val InstructionRegister.pos: Int get() = lsb

inline val InstructionRegister.size_e: Int get() = msb + 1
inline val InstructionRegister.size_i: Int get() = msb - lsb + 1

inline val InstructionRegister.rd: Int get() = (data ushr 11) and 0x1F
inline val InstructionRegister.rt: Int get() = (data ushr 16) and 0x1F
inline val InstructionRegister.rs: Int get() = (data ushr 21) and 0x1F

inline val InstructionRegister.fd: Int get() = (data ushr 6) and 0x1F
inline val InstructionRegister.fs: Int get() = (data ushr 11) and 0x1F
inline val InstructionRegister.ft: Int get() = (data ushr 16) and 0x1F

inline val InstructionRegister.vd: Int get() = (data ushr 0) and 0x7F
inline val InstructionRegister.vs: Int get() = (data ushr 8) and 0x7F
inline val InstructionRegister.vt: Int get() = (data ushr 16) and 0x7F
inline val InstructionRegister.vt1: Int get() = data.extract(0, 1)
inline val InstructionRegister.vt2: Int get() = data.extract(0, 2)
inline val InstructionRegister.vt5: Int get() = data.extract(16, 5)
inline val InstructionRegister.vt5_1: Int get() = vt5 or (vt1 shl 5)
inline val InstructionRegister.vt5_2: Int get() = vt5 or (vt2 shl 5)

inline val InstructionRegister.imm8: Int get() = data.extract(16, 8)
inline val InstructionRegister.imm5: Int get() = data.extract(16, 5)
inline val InstructionRegister.imm3: Int get() = data.extract(16, 3)
inline val InstructionRegister.imm7: Int get() = data.extract(0, 7)
inline val InstructionRegister.imm4: Int get() = data.extract(0, 4)

inline val InstructionRegister.one: Int get() = data.extract(7, 1)
inline val InstructionRegister.two: Int get() = data.extract(15, 1)
inline val InstructionRegister.one_two: Int get() = (1 + 1 * this.one + 2 * this.two)

inline val InstructionRegister.syscall: Int get() = data.extract(6, 20)
inline val InstructionRegister.s_imm16: Int get() = (data shl 16) shr 16
inline val InstructionRegister.u_imm16: Int get() = data and 0xFFFF

inline val InstructionRegister.s_imm14: Int get() = data.extract(2, 14).signExtend(14)

inline val InstructionRegister.u_imm26: Int get() = data.extract(0, 26)
inline val InstructionRegister.jump_address: Int get() = u_imm26 * 4
