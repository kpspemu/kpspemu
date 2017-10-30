@file:Suppress("RedundantUnitReturnType")

package com.soywiz.kpspemu.cpu.interpreter

import com.soywiz.korio.JvmField
import com.soywiz.korio.JvmStatic
import com.soywiz.korio.lang.format
import com.soywiz.korio.util.*
import com.soywiz.kpspemu.cpu.*
import com.soywiz.kpspemu.cpu.dis.disasmMacro
import com.soywiz.kpspemu.util.BitUtils
import com.soywiz.kpspemu.util.imul32_64
import com.soywiz.kpspemu.util.umul32_64
import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.round

// This was discarded because both JVM and JS devirtualize and inline pretty well already
// It can boost things a bit, but not that much. So to keep things simpler, we do not use this for now at least
// Maybe in native that can't devirtualize at runtime we should investigate this again.

/*
@Suppress("RemoveRedundantCallsOfConversionMethods")
class FastCpuInterpreter(var cpu: CpuState, var trace: Boolean = false) {
	fun dispatch(s: CpuState, pc: Int, i: Int): Unit {
		when ((i and 0xFC000000.toInt())) {
			0x00000000.toInt() ->
				when ((i and 0x0000003F.toInt())) {
					0x00000020.toInt() -> return FastInstructionInterpreter.add(s)
					0x00000021.toInt() -> return FastInstructionInterpreter.addu(s)
					0x00000022.toInt() -> return FastInstructionInterpreter.sub(s)
					0x00000023.toInt() -> return FastInstructionInterpreter.subu(s)
					0x00000024.toInt() -> return FastInstructionInterpreter.and(s)
					0x00000027.toInt() -> return FastInstructionInterpreter.nor(s)
					0x00000025.toInt() -> return FastInstructionInterpreter.or(s)
					0x00000026.toInt() -> return FastInstructionInterpreter.xor(s)
					0x00000000.toInt() -> return FastInstructionInterpreter.sll(s)
					0x00000004.toInt() -> return FastInstructionInterpreter.sllv(s)
					0x00000003.toInt() -> return FastInstructionInterpreter.sra(s)
					0x00000007.toInt() -> return FastInstructionInterpreter.srav(s)
					0x00000002.toInt() ->
						when ((i and 0xFFE00000.toInt())) {
							0x00000000.toInt() -> return FastInstructionInterpreter.srl(s)
							0x00200000.toInt() -> return FastInstructionInterpreter.rotr(s)
							else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (0) failed mask 0x%08X".format(i, pc, -2097152))
						}
					0x00000006.toInt() ->
						when ((i and 0xFC0007C0.toInt())) {
							0x00000000.toInt() -> return FastInstructionInterpreter.srlv(s)
							0x00000040.toInt() -> return FastInstructionInterpreter.rotrv(s)
							else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (1) failed mask 0x%08X".format(i, pc, -67106880))
						}
					0x0000002A.toInt() -> return FastInstructionInterpreter.slt(s)
					0x0000002B.toInt() -> return FastInstructionInterpreter.sltu(s)
					0x0000002C.toInt() -> return FastInstructionInterpreter.max(s)
					0x0000002D.toInt() -> return FastInstructionInterpreter.min(s)
					0x0000001A.toInt() -> return FastInstructionInterpreter.div(s)
					0x0000001B.toInt() -> return FastInstructionInterpreter.divu(s)
					0x00000018.toInt() -> return FastInstructionInterpreter.mult(s)
					0x00000019.toInt() -> return FastInstructionInterpreter.multu(s)
					0x0000001C.toInt() -> return FastInstructionInterpreter.madd(s)
					0x0000001D.toInt() -> return FastInstructionInterpreter.maddu(s)
					0x0000002E.toInt() -> return FastInstructionInterpreter.msub(s)
					0x0000002F.toInt() -> return FastInstructionInterpreter.msubu(s)
					0x00000010.toInt() -> return FastInstructionInterpreter.mfhi(s)
					0x00000012.toInt() -> return FastInstructionInterpreter.mflo(s)
					0x00000011.toInt() -> return FastInstructionInterpreter.mthi(s)
					0x00000013.toInt() -> return FastInstructionInterpreter.mtlo(s)
					0x0000000A.toInt() -> return FastInstructionInterpreter.movz(s)
					0x0000000B.toInt() -> return FastInstructionInterpreter.movn(s)
					0x00000016.toInt() -> return FastInstructionInterpreter.clz(s)
					0x00000017.toInt() -> return FastInstructionInterpreter.clo(s)
					0x00000008.toInt() -> return FastInstructionInterpreter.jr(s)
					0x00000009.toInt() -> return FastInstructionInterpreter.jalr(s)
					0x0000000C.toInt() -> return FastInstructionInterpreter.syscall(s)
					0x0000000F.toInt() -> return FastInstructionInterpreter.sync(s)
					0x0000000D.toInt() -> return FastInstructionInterpreter._break(s)
					else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (2) failed mask 0x%08X".format(i, pc, 63))
				}
			0x20000000.toInt() -> return FastInstructionInterpreter.addi(s)
			0x24000000.toInt() -> return FastInstructionInterpreter.addiu(s)
			0x30000000.toInt() -> return FastInstructionInterpreter.andi(s)
			0x34000000.toInt() -> return FastInstructionInterpreter.ori(s)
			0x38000000.toInt() -> return FastInstructionInterpreter.xori(s)
			0x28000000.toInt() -> return FastInstructionInterpreter.slti(s)
			0x2C000000.toInt() -> return FastInstructionInterpreter.sltiu(s)
			0x3C000000.toInt() -> return FastInstructionInterpreter.lui(s)
			0x7C000000.toInt() ->
				when ((i and 0x0000003F.toInt())) {
					0x00000020.toInt() ->
						when ((i and 0xFFE007C0.toInt())) {
							0x7C000400.toInt() -> return FastInstructionInterpreter.seb(s)
							0x7C000600.toInt() -> return FastInstructionInterpreter.seh(s)
							0x7C000500.toInt() -> return FastInstructionInterpreter.bitrev(s)
							0x7C000080.toInt() -> return FastInstructionInterpreter.wsbh(s)
							0x7C0000C0.toInt() -> return FastInstructionInterpreter.wsbw(s)
							else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (3) failed mask 0x%08X".format(i, pc, -2095168))
						}
					0x00000000.toInt() -> return FastInstructionInterpreter.ext(s)
					0x00000004.toInt() -> return FastInstructionInterpreter.ins(s)
					else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (4) failed mask 0x%08X".format(i, pc, 63))
				}
			0x10000000.toInt() -> return FastInstructionInterpreter.beq(s)
			0x50000000.toInt() -> return FastInstructionInterpreter.beql(s)
			0x04000000.toInt() ->
				when ((i and 0x001F0000.toInt())) {
					0x00010000.toInt() -> return FastInstructionInterpreter.bgez(s)
					0x00030000.toInt() -> return FastInstructionInterpreter.bgezl(s)
					0x00110000.toInt() -> return FastInstructionInterpreter.bgezal(s)
					0x00130000.toInt() -> return FastInstructionInterpreter.bgezall(s)
					0x00000000.toInt() -> return FastInstructionInterpreter.bltz(s)
					0x00020000.toInt() -> return FastInstructionInterpreter.bltzl(s)
					0x00100000.toInt() -> return FastInstructionInterpreter.bltzal(s)
					0x00120000.toInt() -> return FastInstructionInterpreter.bltzall(s)
					else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (5) failed mask 0x%08X".format(i, pc, 2031616))
				}
			0x18000000.toInt() -> return FastInstructionInterpreter.blez(s)
			0x58000000.toInt() -> return FastInstructionInterpreter.blezl(s)
			0x1C000000.toInt() -> return FastInstructionInterpreter.bgtz(s)
			0x5C000000.toInt() -> return FastInstructionInterpreter.bgtzl(s)
			0x14000000.toInt() -> return FastInstructionInterpreter.bne(s)
			0x54000000.toInt() -> return FastInstructionInterpreter.bnel(s)
			0x08000000.toInt() -> return FastInstructionInterpreter.j(s)
			0x0C000000.toInt() -> return FastInstructionInterpreter.jal(s)
			0x44000000.toInt() ->
				when ((i and 0x03E00000.toInt())) {
					0x01000000.toInt() ->
						when ((i and 0xFC1F0000.toInt())) {
							0x44000000.toInt() -> return FastInstructionInterpreter.bc1f(s)
							0x44010000.toInt() -> return FastInstructionInterpreter.bc1t(s)
							0x44020000.toInt() -> return FastInstructionInterpreter.bc1fl(s)
							0x44030000.toInt() -> return FastInstructionInterpreter.bc1tl(s)
							else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (6) failed mask 0x%08X".format(i, pc, -65077248))
						}
					0x02000000.toInt() ->
						when ((i and 0xFC00003F.toInt())) {
							0x44000000.toInt() -> return FastInstructionInterpreter.add_s(s)
							0x44000001.toInt() -> return FastInstructionInterpreter.sub_s(s)
							0x44000002.toInt() -> return FastInstructionInterpreter.mul_s(s)
							0x44000003.toInt() -> return FastInstructionInterpreter.div_s(s)
							0x44000004.toInt() -> return FastInstructionInterpreter.sqrt_s(s)
							0x44000005.toInt() -> return FastInstructionInterpreter.abs_s(s)
							0x44000006.toInt() -> return FastInstructionInterpreter.mov_s(s)
							0x44000007.toInt() -> return FastInstructionInterpreter.neg_s(s)
							0x4400000C.toInt() -> return FastInstructionInterpreter.round_w_s(s)
							0x4400000D.toInt() -> return FastInstructionInterpreter.trunc_w_s(s)
							0x4400000E.toInt() -> return FastInstructionInterpreter.ceil_w_s(s)
							0x4400000F.toInt() -> return FastInstructionInterpreter.floor_w_s(s)
							0x44000024.toInt() -> return FastInstructionInterpreter.cvt_w_s(s)
							0x44000030.toInt() -> return FastInstructionInterpreter.c_f_s(s)
							0x44000031.toInt() -> return FastInstructionInterpreter.c_un_s(s)
							0x44000032.toInt() -> return FastInstructionInterpreter.c_eq_s(s)
							0x44000033.toInt() -> return FastInstructionInterpreter.c_ueq_s(s)
							0x44000034.toInt() -> return FastInstructionInterpreter.c_olt_s(s)
							0x44000035.toInt() -> return FastInstructionInterpreter.c_ult_s(s)
							0x44000036.toInt() -> return FastInstructionInterpreter.c_ole_s(s)
							0x44000037.toInt() -> return FastInstructionInterpreter.c_ule_s(s)
							0x44000038.toInt() -> return FastInstructionInterpreter.c_sf_s(s)
							0x44000039.toInt() -> return FastInstructionInterpreter.c_ngle_s(s)
							0x4400003A.toInt() -> return FastInstructionInterpreter.c_seq_s(s)
							0x4400003B.toInt() -> return FastInstructionInterpreter.c_ngl_s(s)
							0x4400003C.toInt() -> return FastInstructionInterpreter.c_lt_s(s)
							0x4400003D.toInt() -> return FastInstructionInterpreter.c_nge_s(s)
							0x4400003E.toInt() -> return FastInstructionInterpreter.c_le_s(s)
							0x4400003F.toInt() -> return FastInstructionInterpreter.c_ngt_s(s)
							else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (7) failed mask 0x%08X".format(i, pc, -67108801))
						}
					0x02800000.toInt() -> return FastInstructionInterpreter.cvt_s_w(s)
					0x00000000.toInt() -> return FastInstructionInterpreter.mfc1(s)
					0x00800000.toInt() -> return FastInstructionInterpreter.mtc1(s)
					0x00400000.toInt() -> return FastInstructionInterpreter.cfc1(s)
					0x00C00000.toInt() -> return FastInstructionInterpreter.ctc1(s)
					else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (8) failed mask 0x%08X".format(i, pc, 65011712))
				}
			0x80000000.toInt() -> return FastInstructionInterpreter.lb(s)
			0x84000000.toInt() -> return FastInstructionInterpreter.lh(s)
			0x8C000000.toInt() -> return FastInstructionInterpreter.lw(s)
			0x88000000.toInt() -> return FastInstructionInterpreter.lwl(s)
			0x98000000.toInt() -> return FastInstructionInterpreter.lwr(s)
			0x90000000.toInt() -> return FastInstructionInterpreter.lbu(s)
			0x94000000.toInt() -> return FastInstructionInterpreter.lhu(s)
			0xA0000000.toInt() -> return FastInstructionInterpreter.sb(s)
			0xA4000000.toInt() -> return FastInstructionInterpreter.sh(s)
			0xAC000000.toInt() -> return FastInstructionInterpreter.sw(s)
			0xA8000000.toInt() -> return FastInstructionInterpreter.swl(s)
			0xB8000000.toInt() -> return FastInstructionInterpreter.swr(s)
			0xC0000000.toInt() -> return FastInstructionInterpreter.ll(s)
			0xE0000000.toInt() -> return FastInstructionInterpreter.sc(s)
			0xC4000000.toInt() -> return FastInstructionInterpreter.lwc1(s)
			0xE4000000.toInt() -> return FastInstructionInterpreter.swc1(s)
			0xBC000000.toInt() -> return FastInstructionInterpreter.cache(s)
			0x70000000.toInt() ->
				when ((i and 0x000007FF.toInt())) {
					0x0000003F.toInt() -> return FastInstructionInterpreter.dbreak(s)
					0x00000000.toInt() -> return FastInstructionInterpreter.halt(s)
					0x0000003E.toInt() -> return FastInstructionInterpreter.dret(s)
					0x00000024.toInt() -> return FastInstructionInterpreter.mfic(s)
					0x00000026.toInt() -> return FastInstructionInterpreter.mtic(s)
					0x0000003D.toInt() ->
						when ((i and 0xFFE00000.toInt())) {
							0x70000000.toInt() -> return FastInstructionInterpreter.mfdr(s)
							0x70800000.toInt() -> return FastInstructionInterpreter.mtdr(s)
							else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (9) failed mask 0x%08X".format(i, pc, -2097152))
						}
					else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (10) failed mask 0x%08X".format(i, pc, 2047))
				}
			0x40000000.toInt() ->
				when ((i and 0x03E007FF.toInt())) {
					0x02000018.toInt() -> return FastInstructionInterpreter.eret(s)
					0x00400000.toInt() -> return FastInstructionInterpreter.cfc0(s)
					0x00C00000.toInt() -> return FastInstructionInterpreter.ctc0(s)
					0x00000000.toInt() -> return FastInstructionInterpreter.mfc0(s)
					0x00800000.toInt() -> return FastInstructionInterpreter.mtc0(s)
					else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (11) failed mask 0x%08X".format(i, pc, 65013759))
				}
			0x48000000.toInt() ->
				when ((i and 0x03E00000.toInt())) {
					0x00600000.toInt() ->
						when ((i and 0xFC00FF80.toInt())) {
							0x48000000.toInt() -> return FastInstructionInterpreter.mfv(s)
							0x48000080.toInt() -> return FastInstructionInterpreter.mfvc(s)
							else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (12) failed mask 0x%08X".format(i, pc, -67043456))
						}
					0x00E00000.toInt() ->
						when ((i and 0xFC00FF80.toInt())) {
							0x48000000.toInt() -> return FastInstructionInterpreter.mtv(s)
							0x48000080.toInt() -> return FastInstructionInterpreter.mtvc(s)
							else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (13) failed mask 0x%08X".format(i, pc, -67043456))
						}
					0x01000000.toInt() ->
						when ((i and 0xFC030000.toInt())) {
							0x48000000.toInt() -> return FastInstructionInterpreter.bvf(s)
							0x48010000.toInt() -> return FastInstructionInterpreter.bvt(s)
							0x48020000.toInt() -> return FastInstructionInterpreter.bvfl(s)
							0x48030000.toInt() -> return FastInstructionInterpreter.bvtl(s)
							else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (14) failed mask 0x%08X".format(i, pc, -66912256))
						}
					else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (15) failed mask 0x%08X".format(i, pc, 65011712))
				}
			0xC8000000.toInt() -> return FastInstructionInterpreter.lv_s(s)
			0xD8000000.toInt() -> return FastInstructionInterpreter.lv_q(s)
			0xD4000000.toInt() ->
				when ((i and 0x00000002.toInt())) {
					0x00000000.toInt() -> return FastInstructionInterpreter.lvl_q(s)
					0x00000002.toInt() -> return FastInstructionInterpreter.lvr_q(s)
					else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (16) failed mask 0x%08X".format(i, pc, 2))
				}
			0xF8000000.toInt() -> return FastInstructionInterpreter.sv_q(s)
			0x64000000.toInt() ->
				when ((i and 0x03800000.toInt())) {
					0x00800000.toInt() -> return FastInstructionInterpreter.vdot(s)
					0x01000000.toInt() -> return FastInstructionInterpreter.vscl(s)
					0x02000000.toInt() -> return FastInstructionInterpreter.vhdp(s)
					0x02800000.toInt() -> return FastInstructionInterpreter.vcrs_t(s)
					0x00000000.toInt() -> return FastInstructionInterpreter.vmul(s)
					0x03000000.toInt() -> return FastInstructionInterpreter.vdet(s)
					else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (17) failed mask 0x%08X".format(i, pc, 58720256))
				}
			0x6C000000.toInt() ->
				when ((i and 0x03800000.toInt())) {
					0x03000000.toInt() -> return FastInstructionInterpreter.vsge(s)
					0x03800000.toInt() -> return FastInstructionInterpreter.vslt(s)
					0x01000000.toInt() -> return FastInstructionInterpreter.vmin(s)
					0x01800000.toInt() -> return FastInstructionInterpreter.vmax(s)
					0x00000000.toInt() -> return FastInstructionInterpreter.vcmp(s)
					0x02800000.toInt() -> return FastInstructionInterpreter.vscmp(s)
					else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (18) failed mask 0x%08X".format(i, pc, 58720256))
				}
			0xF0000000.toInt() ->
				when ((i and 0x03800000.toInt())) {
					0x03800000.toInt() ->
						when ((i and 0xFC600000.toInt())) {
							0xF0200000.toInt() -> return FastInstructionInterpreter.vrot(s)
							0xF0000000.toInt() ->
								when ((i and 0x039F0000.toInt())) {
									0x03830000.toInt() -> return FastInstructionInterpreter.vmidt(s)
									0x03800000.toInt() -> return FastInstructionInterpreter.vmmov(s)
									0x03860000.toInt() -> return FastInstructionInterpreter.vmzero(s)
									0x03870000.toInt() -> return FastInstructionInterpreter.vmone(s)
									else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (19) failed mask 0x%08X".format(i, pc, 60751872))
								}
							else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (20) failed mask 0x%08X".format(i, pc, -60817408))
						}
					0x00000000.toInt() -> return FastInstructionInterpreter.vmmul(s)
					0x02800000.toInt() ->
						when ((i and 0xFC008080.toInt())) {
							0xF0008000.toInt() -> return FastInstructionInterpreter.vcrsp_t(s)
							0xF0008080.toInt() -> return FastInstructionInterpreter.vqmul(s)
							else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (21) failed mask 0x%08X".format(i, pc, -67075968))
						}
					0x00800000.toInt() ->
						when ((i and 0xFC008080.toInt())) {
							0xF0000080.toInt() -> return FastInstructionInterpreter.vtfm2(s)
							0xF0000000.toInt() -> return FastInstructionInterpreter.vhtfm2(s)
							else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (22) failed mask 0x%08X".format(i, pc, -67075968))
						}
					0x01000000.toInt() ->
						when ((i and 0xFC008080.toInt())) {
							0xF0008000.toInt() -> return FastInstructionInterpreter.vtfm3(s)
							0xF0000080.toInt() -> return FastInstructionInterpreter.vhtfm3(s)
							else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (23) failed mask 0x%08X".format(i, pc, -67075968))
						}
					0x01800000.toInt() ->
						when ((i and 0xFC008080.toInt())) {
							0xF0008080.toInt() -> return FastInstructionInterpreter.vtfm4(s)
							0xF0008000.toInt() -> return FastInstructionInterpreter.vhtfm4(s)
							else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (24) failed mask 0x%08X".format(i, pc, -67075968))
						}
					0x02000000.toInt() -> return FastInstructionInterpreter.vmscl(s)
					else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (25) failed mask 0x%08X".format(i, pc, 58720256))
				}
			0xD0000000.toInt() ->
				when ((i and 0x03000000.toInt())) {
					0x00000000.toInt() ->
						when ((i and 0xFCE00000.toInt())) {
							0xD0000000.toInt() ->
								when ((i and 0x031F0000.toInt())) {
									0x00060000.toInt() -> return FastInstructionInterpreter.vzero(s)
									0x00070000.toInt() -> return FastInstructionInterpreter.vone(s)
									0x00000000.toInt() -> return FastInstructionInterpreter.vmov(s)
									0x00010000.toInt() -> return FastInstructionInterpreter.vabs(s)
									0x00020000.toInt() -> return FastInstructionInterpreter.vneg(s)
									0x00100000.toInt() -> return FastInstructionInterpreter.vrcp(s)
									0x00110000.toInt() -> return FastInstructionInterpreter.vrsq(s)
									0x00120000.toInt() -> return FastInstructionInterpreter.vsin(s)
									0x00130000.toInt() -> return FastInstructionInterpreter.vcos(s)
									0x00140000.toInt() -> return FastInstructionInterpreter.vexp2(s)
									0x00150000.toInt() -> return FastInstructionInterpreter.vlog2(s)
									0x00160000.toInt() -> return FastInstructionInterpreter.vsqrt(s)
									0x00170000.toInt() -> return FastInstructionInterpreter.vasin(s)
									0x00180000.toInt() -> return FastInstructionInterpreter.vnrcp(s)
									0x001A0000.toInt() -> return FastInstructionInterpreter.vnsin(s)
									0x001C0000.toInt() -> return FastInstructionInterpreter.vrexp2(s)
									0x00040000.toInt() -> return FastInstructionInterpreter.vsat0(s)
									0x00050000.toInt() -> return FastInstructionInterpreter.vsat1(s)
									0x00030000.toInt() -> return FastInstructionInterpreter.vidt(s)
									else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (26) failed mask 0x%08X".format(i, pc, 52363264))
								}
							0xD0400000.toInt() ->
								when ((i and 0x031F0000.toInt())) {
									0x00040000.toInt() -> return FastInstructionInterpreter.vocp(s)
									0x000A0000.toInt() -> return FastInstructionInterpreter.vsgn(s)
									0x00080000.toInt() -> return FastInstructionInterpreter.vsrt3(s)
									0x00060000.toInt() -> return FastInstructionInterpreter.vfad(s)
									0x00070000.toInt() -> return FastInstructionInterpreter.vavg(s)
									0x00190000.toInt() -> return FastInstructionInterpreter.vt4444_q(s)
									0x001A0000.toInt() -> return FastInstructionInterpreter.vt5551_q(s)
									0x001B0000.toInt() -> return FastInstructionInterpreter.vt5650_q(s)
									0x00100000.toInt() -> return FastInstructionInterpreter.vmfvc(s)
									0x00110000.toInt() -> return FastInstructionInterpreter.vmtvc(s)
									0x00020000.toInt() -> return FastInstructionInterpreter.vbfy1(s)
									0x00030000.toInt() -> return FastInstructionInterpreter.vbfy2(s)
									0x00050000.toInt() -> return FastInstructionInterpreter.vsocp(s)
									0x00000000.toInt() -> return FastInstructionInterpreter.vsrt1(s)
									0x00010000.toInt() -> return FastInstructionInterpreter.vsrt2(s)
									0x00090000.toInt() -> return FastInstructionInterpreter.vsrt4(s)
									else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (27) failed mask 0x%08X".format(i, pc, 52363264))
								}
							0xD0600000.toInt() -> return FastInstructionInterpreter.vcst(s)
							0xD0200000.toInt() ->
								when ((i and 0x031F0000.toInt())) {
									0x001D0000.toInt() -> return FastInstructionInterpreter.vi2c(s)
									0x001C0000.toInt() -> return FastInstructionInterpreter.vi2uc(s)
									0x00000000.toInt() -> return FastInstructionInterpreter.vrnds(s)
									0x00010000.toInt() -> return FastInstructionInterpreter.vrndi(s)
									0x00020000.toInt() -> return FastInstructionInterpreter.vrndf1(s)
									0x00030000.toInt() -> return FastInstructionInterpreter.vrndf2(s)
									0x00120000.toInt() -> return FastInstructionInterpreter.vf2h(s)
									0x00130000.toInt() -> return FastInstructionInterpreter.vh2f(s)
									0x001F0000.toInt() -> return FastInstructionInterpreter.vi2s(s)
									0x001E0000.toInt() -> return FastInstructionInterpreter.vi2us(s)
									0x00170000.toInt() -> return FastInstructionInterpreter.vlgb(s)
									0x001B0000.toInt() -> return FastInstructionInterpreter.vs2i(s)
									0x00190000.toInt() -> return FastInstructionInterpreter.vc2i(s)
									0x00180000.toInt() -> return FastInstructionInterpreter.vuc2i(s)
									0x00160000.toInt() -> return FastInstructionInterpreter.vsbz(s)
									0x001A0000.toInt() -> return FastInstructionInterpreter.vus2i(s)
									else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (28) failed mask 0x%08X".format(i, pc, 52363264))
								}
							else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (29) failed mask 0x%08X".format(i, pc, -52428800))
						}
					0x02000000.toInt() ->
						when ((i and 0xFCE00000.toInt())) {
							0xD0A00000.toInt() ->
								when ((i and 0x03180000.toInt())) {
									0x02080000.toInt() -> return FastInstructionInterpreter.vcmovf(s)
									0x02000000.toInt() -> return FastInstructionInterpreter.vcmovt(s)
									else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (30) failed mask 0x%08X".format(i, pc, 51904512))
								}
							0xD0600000.toInt() -> return FastInstructionInterpreter.vf2id(s)
							0xD0000000.toInt() -> return FastInstructionInterpreter.vf2in(s)
							0xD0400000.toInt() -> return FastInstructionInterpreter.vf2iu(s)
							0xD0200000.toInt() -> return FastInstructionInterpreter.vf2iz(s)
							0xD0800000.toInt() -> return FastInstructionInterpreter.vi2f(s)
							else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (31) failed mask 0x%08X".format(i, pc, -52428800))
						}
					0x03000000.toInt() -> return FastInstructionInterpreter.vwbn(s)
					else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (32) failed mask 0x%08X".format(i, pc, 50331648))
				}
			0x60000000.toInt() ->
				when ((i and 0x03800000.toInt())) {
					0x00000000.toInt() -> return FastInstructionInterpreter.vadd(s)
					0x00800000.toInt() -> return FastInstructionInterpreter.vsub(s)
					0x03800000.toInt() -> return FastInstructionInterpreter.vdiv(s)
					0x01000000.toInt() -> return FastInstructionInterpreter.vsbn(s)
					else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (33) failed mask 0x%08X".format(i, pc, 58720256))
				}
			0xDC000000.toInt() ->
				when ((i and 0x03000000.toInt())) {
					0x03000000.toInt() ->
						when ((i and 0xFC800000.toInt())) {
							0xDC000000.toInt() -> return FastInstructionInterpreter.viim(s)
							0xDC800000.toInt() -> return FastInstructionInterpreter.vfim(s)
							else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (34) failed mask 0x%08X".format(i, pc, -58720256))
						}
					0x02000000.toInt() -> return FastInstructionInterpreter.vpfxd(s)
					0x00000000.toInt() -> return FastInstructionInterpreter.vpfxs(s)
					0x01000000.toInt() -> return FastInstructionInterpreter.vpfxt(s)
					else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (35) failed mask 0x%08X".format(i, pc, 50331648))
				}
			0xFC000000.toInt() ->
				when ((i and 0x03FFFFFF.toInt())) {
					0x03FF0000.toInt() -> return FastInstructionInterpreter.vnop(s)
					0x03FF0320.toInt() -> return FastInstructionInterpreter.vsync(s)
					0x03FF040D.toInt() -> return FastInstructionInterpreter.vflush(s)
					else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (36) failed mask 0x%08X".format(i, pc, 67108863))
				}
			0x68000000.toInt() -> return FastInstructionInterpreter.mfvme(s)
			0xB0000000.toInt() -> return FastInstructionInterpreter.mtvme(s)
			0xE8000000.toInt() -> return FastInstructionInterpreter.sv_s(s)
			0xF4000000.toInt() ->
				when ((i and 0x00000002.toInt())) {
					0x00000000.toInt() -> return FastInstructionInterpreter.svl_q(s)
					0x00000002.toInt() -> return FastInstructionInterpreter.svr_q(s)
					else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (37) failed mask 0x%08X".format(i, pc, 2))
				}
			else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (38) failed mask 0x%08X".format(i, pc, -67108864))
		}
	}

	fun steps(count: Int) {
		val cpu = this.cpu
		val mem = cpu.mem
		for (n in 0 until count) {
			val PC = cpu._PC
			if (PC == 0) throw IllegalStateException("Trying to execute PC=0")
			if (trace) println("%08X: %s".format(PC, cpu.mem.disasmMacro(PC)))
			val IR = mem.lw(PC)
			cpu.IR = IR
			dispatch(cpu, PC, IR)
		}
	}
}

// http://www.mrc.uidaho.edu/mrc/people/jff/digital/MIPSir.html
object FastInstructionInterpreter : InstructionDecoder() {
	@JvmField
	val itemp = IntArray(2)

	@JvmStatic
	fun unimplemented(s: CpuState, i: InstructionType): Unit = TODO("unimplemented: ${i.name} : " + i + " at ${"%08X".format(s._PC)}")

	// ALU
	@JvmStatic
	fun lui(s: CpuState) = s { RT = (U_IMM16 shl 16) }

	@JvmStatic
	fun movz(s: CpuState) = s { if (RT == 0) RD = RS }

	@JvmStatic
	fun movn(s: CpuState) = s { if (RT != 0) RD = RS }

	@JvmStatic
	fun ext(s: CpuState) = s { RT = RS.extract(POS, SIZE_E) }

	@JvmStatic
	fun ins(s: CpuState) = s { RT = RT.insert(RS, POS, SIZE_I) }

	@JvmStatic
	fun clz(s: CpuState) = s { RD = BitUtils.clz(RS) }

	@JvmStatic
	fun clo(s: CpuState) = s { RD = BitUtils.clo(RS) }

	@JvmStatic
	fun seb(s: CpuState) = s { RD = BitUtils.seb(RT) }

	@JvmStatic
	fun seh(s: CpuState) = s { RD = BitUtils.seh(RT) }

	@JvmStatic
	fun wsbh(s: CpuState) = s { RD = BitUtils.wsbh(RT) }

	@JvmStatic
	fun wsbw(s: CpuState) = s { RD = BitUtils.wsbw(RT) }

	@JvmStatic
	fun max(s: CpuState) = s { RD = kotlin.math.max(RS, RT) }

	@JvmStatic
	fun min(s: CpuState) = s { RD = kotlin.math.min(RS, RT) }

	@JvmStatic
	fun add(s: CpuState) = s { RD = RS + RT }

	@JvmStatic
	fun addu(s: CpuState) = s { RD = RS + RT }

	@JvmStatic
	fun sub(s: CpuState) = s { RD = RS - RT }

	@JvmStatic
	fun subu(s: CpuState) = s { RD = RS - RT }

	@JvmStatic
	fun addi(s: CpuState) = s { RT = RS + S_IMM16 }

	@JvmStatic
	fun addiu(s: CpuState) = s { RT = RS + S_IMM16 }

	@JvmStatic
	fun div(s: CpuState) = s { LO = RS / RT; HI = RS % RT }

	@JvmStatic
	fun divu(s: CpuState) = s { LO = RS udiv RT; HI = RS urem RT }

	@JvmStatic
	fun mult(s: CpuState) = s {
		imul32_64(RS, RT, itemp)
		this.LO = itemp[0]
		this.HI = itemp[1]
	}

	@JvmStatic
	fun multu(s: CpuState) = s {
		umul32_64(RS, RT, itemp)
		this.LO = itemp[0]
		this.HI = itemp[1]
	}

	@JvmStatic
	fun mflo(s: CpuState) = s { RD = LO }

	@JvmStatic
	fun mfhi(s: CpuState) = s { RD = HI }

	@JvmStatic
	fun mfic(s: CpuState) = s { RT = IC }

	@JvmStatic
	fun mtlo(s: CpuState) = s { LO = RS }

	@JvmStatic
	fun mthi(s: CpuState) = s { HI = RS }

	@JvmStatic
	fun mtic(s: CpuState) = s { IC = RT }

	// ALU: Bit
	@JvmStatic
	fun or(s: CpuState) = s { RD = RS or RT }

	@JvmStatic
	fun xor(s: CpuState) = s { RD = RS xor RT }

	@JvmStatic
	fun and(s: CpuState) = s { RD = RS and RT }

	@JvmStatic
	fun nor(s: CpuState) = s { RD = (RS or RT).inv() }

	@JvmStatic
	fun ori(s: CpuState) = s { RT = RS or U_IMM16 }

	@JvmStatic
	fun xori(s: CpuState) = s { RT = RS xor U_IMM16 }

	@JvmStatic
	fun andi(s: CpuState) = s { RT = RS and U_IMM16 }

	@JvmStatic
	fun sll(s: CpuState) = s { RD = RT shl POS }

	@JvmStatic
	fun sra(s: CpuState) = s { RD = RT shr POS }

	@JvmStatic
	fun srl(s: CpuState) = s { RD = RT ushr POS }

	@JvmStatic
	fun sllv(s: CpuState) = s { RD = RT shl (RS and 0b11111) }

	@JvmStatic
	fun srav(s: CpuState) = s { RD = RT shr (RS and 0b11111) }

	@JvmStatic
	fun srlv(s: CpuState) = s { RD = RT ushr (RS and 0b11111) }

	// Memory
	@JvmStatic
	fun lb(s: CpuState) = s { RT = mem.lb(RS_IMM16) }

	@JvmStatic
	fun lbu(s: CpuState) = s { RT = mem.lbu(RS_IMM16) }

	@JvmStatic
	fun lh(s: CpuState) = s { RT = mem.lh(RS_IMM16) }

	@JvmStatic
	fun lhu(s: CpuState) = s { RT = mem.lhu(RS_IMM16) }

	@JvmStatic
	fun lw(s: CpuState) = s { RT = mem.lw(RS_IMM16) }

	@JvmStatic
	fun sb(s: CpuState) = s { mem.sb(RS_IMM16, RT) }

	@JvmStatic
	fun sh(s: CpuState) = s { mem.sh(RS_IMM16, RT) }

	@JvmStatic
	fun sw(s: CpuState) = s { mem.sw(RS_IMM16, RT) }

	@JvmStatic
	fun lwc1(s: CpuState) = s { FT_I = mem.lw(RS_IMM16) }

	@JvmStatic
	fun swc1(s: CpuState) = s { mem.sw(RS_IMM16, FT_I) }

	// Special
	@JvmStatic
	fun syscall(s: CpuState) = s.preadvance { syscall(SYSCALL) }

	@JvmStatic
	fun _break(s: CpuState) = s.preadvance { throw CpuBreak(SYSCALL) }

	// Set less
	@JvmStatic
	fun slt(s: CpuState) = s { RD = if (IntEx.compare(RS, RT) < 0) 1 else 0 }

	@JvmStatic
	fun sltu(s: CpuState) = s { RD = if (IntEx.compareUnsigned(RS, RT) < 0) 1 else 0 }

	@JvmStatic
	fun slti(s: CpuState) = s { RT = if (IntEx.compare(RS, S_IMM16) < 0) 1 else 0 }

	@JvmStatic
	fun sltiu(s: CpuState) = s { RT = if (IntEx.compareUnsigned(RS, S_IMM16) < 0) 1 else 0 }


	// Branch
	@JvmStatic
	fun beq(s: CpuState) = s.branch { RS == RT }

	@JvmStatic
	fun bne(s: CpuState) = s.branch { RS != RT }

	@JvmStatic
	fun bltz(s: CpuState) = s.branch { RS < 0 }

	@JvmStatic
	fun blez(s: CpuState) = s.branch { RS <= 0 }

	@JvmStatic
	fun bgtz(s: CpuState) = s.branch { RS > 0 }

	@JvmStatic
	fun bgez(s: CpuState) = s.branch { RS >= 0 }

	@JvmStatic
	fun beql(s: CpuState) = s.branchLikely { RS == RT }

	@JvmStatic
	fun bnel(s: CpuState) = s.branchLikely { RS != RT }

	@JvmStatic
	fun bltzl(s: CpuState) = s.branchLikely { RS < 0 }

	@JvmStatic
	fun blezl(s: CpuState) = s.branchLikely { RS <= 0 }

	@JvmStatic
	fun bgtzl(s: CpuState) = s.branchLikely { RS > 0 }

	@JvmStatic
	fun bgezl(s: CpuState) = s.branchLikely { RS >= 0 }

	@JvmStatic
	fun j(s: CpuState) = s.none { _PC = _nPC; _nPC = (_PC and 0xf0000000.toInt()) or (JUMP_ADDRESS) }

	@JvmStatic
	fun jr(s: CpuState) = s.none { _PC = _nPC; _nPC = RS }

	// $31 = PC + 8 (or nPC + 4); PC = nPC; nPC = (PC & 0xf0000000) | (target << 2);
	@JvmStatic
	fun jal(s: CpuState) = s.none { RA = _nPC + 4; j(s) }

	@JvmStatic
	fun jalr(s: CpuState) = s.none { RA = _nPC + 4; jr(s) }

	// Float
	@JvmStatic
	fun mfc1(s: CpuState) = s { RT = FS_I }

	@JvmStatic
	fun mtc1(s: CpuState) = s { FS_I = RT }

	@JvmStatic
	fun cvt_s_w(s: CpuState) = s { FD = FS_I.toFloat() }

	@JvmStatic
	fun cvt_w_s(s: CpuState) = s {
		// @TODO: _cvt_w_s_impl, fcr31_rm: 0:rint, 1:cast, 2:ceil, 3:floor
		FD_I = FS.toInt()
	}

	@JvmStatic
	fun trunc_w_s(s: CpuState) = s { FD_I = FS.toInt() }

	@JvmStatic
	fun round_w_s(s: CpuState) = s { FD_I = round(FS).toInt() }

	@JvmStatic
	fun ceil_w_s(s: CpuState) = s { FD_I = ceil(FS).toInt() }

	@JvmStatic
	fun floor_w_s(s: CpuState) = s { FD_I = floor(FS).toInt() }

	@JvmStatic
	fun mov_s(s: CpuState) = s { FD = FS }

	@JvmStatic
	fun add_s(s: CpuState) = s { FD = FS + FT }

	@JvmStatic
	fun sub_s(s: CpuState) = s { FD = FS - FT }

	@JvmStatic
	fun mul_s(s: CpuState) = s { FD = FS * FT }

	@JvmStatic
	fun div_s(s: CpuState) = s { FD = FS / FT }

	@JvmStatic
	fun neg_s(s: CpuState) = s { FD = -FS }

	@JvmStatic
	fun abs_s(s: CpuState) = s { FD = kotlin.math.abs(FS) }

	@JvmStatic
	fun sqrt_s(s: CpuState) = s { FD = kotlin.math.sqrt(FS) }

	// Missing
	@JvmStatic
	fun bitrev(s: CpuState) = unimplemented(s, Instructions.bitrev)

	@JvmStatic
	fun rotr(s: CpuState) = unimplemented(s, Instructions.rotr)

	@JvmStatic
	fun rotrv(s: CpuState) = unimplemented(s, Instructions.rotrv)

	@JvmStatic
	fun madd(s: CpuState): Unit = unimplemented(s, Instructions.madd)

	@JvmStatic
	fun maddu(s: CpuState): Unit = unimplemented(s, Instructions.maddu)

	@JvmStatic
	fun msub(s: CpuState): Unit = unimplemented(s, Instructions.msub)

	@JvmStatic
	fun msubu(s: CpuState): Unit = unimplemented(s, Instructions.msubu)

	@JvmStatic
	fun bgezal(s: CpuState): Unit = unimplemented(s, Instructions.bgezal)

	@JvmStatic
	fun bgezall(s: CpuState): Unit = unimplemented(s, Instructions.bgezall)

	@JvmStatic
	fun bltzal(s: CpuState): Unit = unimplemented(s, Instructions.bltzal)

	@JvmStatic
	fun bltzall(s: CpuState): Unit = unimplemented(s, Instructions.bltzall)

	@JvmStatic
	fun bc1f(s: CpuState): Unit = unimplemented(s, Instructions.bc1f)

	@JvmStatic
	fun bc1t(s: CpuState): Unit = unimplemented(s, Instructions.bc1t)

	@JvmStatic
	fun bc1fl(s: CpuState): Unit = unimplemented(s, Instructions.bc1fl)

	@JvmStatic
	fun bc1tl(s: CpuState): Unit = unimplemented(s, Instructions.bc1tl)

	@JvmStatic
	fun lwl(s: CpuState): Unit = unimplemented(s, Instructions.lwl)

	@JvmStatic
	fun lwr(s: CpuState): Unit = unimplemented(s, Instructions.lwr)

	@JvmStatic
	fun swl(s: CpuState): Unit = unimplemented(s, Instructions.swl)

	@JvmStatic
	fun swr(s: CpuState): Unit = unimplemented(s, Instructions.swr)

	@JvmStatic
	fun ll(s: CpuState): Unit = unimplemented(s, Instructions.ll)

	@JvmStatic
	fun sc(s: CpuState): Unit = unimplemented(s, Instructions.sc)

	@JvmStatic
	fun cfc1(s: CpuState): Unit = unimplemented(s, Instructions.cfc1)

	@JvmStatic
	fun ctc1(s: CpuState): Unit = unimplemented(s, Instructions.ctc1)

	@JvmStatic
	fun c_f_s(s: CpuState): Unit = unimplemented(s, Instructions.c_f_s)

	@JvmStatic
	fun c_un_s(s: CpuState): Unit = unimplemented(s, Instructions.c_un_s)

	@JvmStatic
	fun c_eq_s(s: CpuState): Unit = unimplemented(s, Instructions.c_eq_s)

	@JvmStatic
	fun c_ueq_s(s: CpuState): Unit = unimplemented(s, Instructions.c_ueq_s)

	@JvmStatic
	fun c_olt_s(s: CpuState): Unit = unimplemented(s, Instructions.c_olt_s)

	@JvmStatic
	fun c_ult_s(s: CpuState): Unit = unimplemented(s, Instructions.c_ult_s)

	@JvmStatic
	fun c_ole_s(s: CpuState): Unit = unimplemented(s, Instructions.c_ole_s)

	@JvmStatic
	fun c_ule_s(s: CpuState): Unit = unimplemented(s, Instructions.c_ule_s)

	@JvmStatic
	fun c_sf_s(s: CpuState): Unit = unimplemented(s, Instructions.c_sf_s)

	@JvmStatic
	fun c_ngle_s(s: CpuState): Unit = unimplemented(s, Instructions.c_ngle_s)

	@JvmStatic
	fun c_seq_s(s: CpuState): Unit = unimplemented(s, Instructions.c_seq_s)

	@JvmStatic
	fun c_ngl_s(s: CpuState): Unit = unimplemented(s, Instructions.c_ngl_s)

	@JvmStatic
	fun c_lt_s(s: CpuState): Unit = unimplemented(s, Instructions.c_lt_s)

	@JvmStatic
	fun c_nge_s(s: CpuState): Unit = unimplemented(s, Instructions.c_nge_s)

	@JvmStatic
	fun c_le_s(s: CpuState): Unit = unimplemented(s, Instructions.c_le_s)

	@JvmStatic
	fun c_ngt_s(s: CpuState): Unit = unimplemented(s, Instructions.c_ngt_s)

	@JvmStatic
	fun cache(s: CpuState): Unit = unimplemented(s, Instructions.cache)

	@JvmStatic
	fun sync(s: CpuState): Unit = unimplemented(s, Instructions.sync)

	@JvmStatic
	fun dbreak(s: CpuState): Unit = unimplemented(s, Instructions.dbreak)

	@JvmStatic
	fun halt(s: CpuState): Unit = unimplemented(s, Instructions.halt)

	@JvmStatic
	fun dret(s: CpuState): Unit = unimplemented(s, Instructions.dret)

	@JvmStatic
	fun eret(s: CpuState): Unit = unimplemented(s, Instructions.eret)

	@JvmStatic
	fun mfdr(s: CpuState): Unit = unimplemented(s, Instructions.mfdr)

	@JvmStatic
	fun mtdr(s: CpuState): Unit = unimplemented(s, Instructions.mtdr)

	@JvmStatic
	fun cfc0(s: CpuState): Unit = unimplemented(s, Instructions.cfc0)

	@JvmStatic
	fun ctc0(s: CpuState): Unit = unimplemented(s, Instructions.ctc0)

	@JvmStatic
	fun mfc0(s: CpuState): Unit = unimplemented(s, Instructions.mfc0)

	@JvmStatic
	fun mtc0(s: CpuState): Unit = unimplemented(s, Instructions.mtc0)

	@JvmStatic
	fun mfv(s: CpuState): Unit = unimplemented(s, Instructions.mfv)

	@JvmStatic
	fun mfvc(s: CpuState): Unit = unimplemented(s, Instructions.mfvc)

	@JvmStatic
	fun mtv(s: CpuState): Unit = unimplemented(s, Instructions.mtv)

	@JvmStatic
	fun mtvc(s: CpuState): Unit = unimplemented(s, Instructions.mtvc)

	@JvmStatic
	fun lv_s(s: CpuState): Unit = unimplemented(s, Instructions.lv_s)

	@JvmStatic
	fun lv_q(s: CpuState): Unit = unimplemented(s, Instructions.lv_q)

	@JvmStatic
	fun lvl_q(s: CpuState): Unit = unimplemented(s, Instructions.lvl_q)

	@JvmStatic
	fun lvr_q(s: CpuState): Unit = unimplemented(s, Instructions.lvr_q)

	@JvmStatic
	fun sv_q(s: CpuState): Unit = unimplemented(s, Instructions.sv_q)

	@JvmStatic
	fun vdot(s: CpuState): Unit = unimplemented(s, Instructions.vdot)

	@JvmStatic
	fun vscl(s: CpuState): Unit = unimplemented(s, Instructions.vscl)

	@JvmStatic
	fun vsge(s: CpuState): Unit = unimplemented(s, Instructions.vsge)

	@JvmStatic
	fun vslt(s: CpuState): Unit = unimplemented(s, Instructions.vslt)

	@JvmStatic
	fun vrot(s: CpuState): Unit = unimplemented(s, Instructions.vrot)

	@JvmStatic
	fun vzero(s: CpuState): Unit = unimplemented(s, Instructions.vzero)

	@JvmStatic
	fun vone(s: CpuState): Unit = unimplemented(s, Instructions.vone)

	@JvmStatic
	fun vmov(s: CpuState): Unit = unimplemented(s, Instructions.vmov)

	@JvmStatic
	fun vabs(s: CpuState): Unit = unimplemented(s, Instructions.vabs)

	@JvmStatic
	fun vneg(s: CpuState): Unit = unimplemented(s, Instructions.vneg)

	@JvmStatic
	fun vocp(s: CpuState): Unit = unimplemented(s, Instructions.vocp)

	@JvmStatic
	fun vsgn(s: CpuState): Unit = unimplemented(s, Instructions.vsgn)

	@JvmStatic
	fun vrcp(s: CpuState): Unit = unimplemented(s, Instructions.vrcp)

	@JvmStatic
	fun vrsq(s: CpuState): Unit = unimplemented(s, Instructions.vrsq)

	@JvmStatic
	fun vsin(s: CpuState): Unit = unimplemented(s, Instructions.vsin)

	@JvmStatic
	fun vcos(s: CpuState): Unit = unimplemented(s, Instructions.vcos)

	@JvmStatic
	fun vexp2(s: CpuState): Unit = unimplemented(s, Instructions.vexp2)

	@JvmStatic
	fun vlog2(s: CpuState): Unit = unimplemented(s, Instructions.vlog2)

	@JvmStatic
	fun vsqrt(s: CpuState): Unit = unimplemented(s, Instructions.vsqrt)

	@JvmStatic
	fun vasin(s: CpuState): Unit = unimplemented(s, Instructions.vasin)

	@JvmStatic
	fun vnrcp(s: CpuState): Unit = unimplemented(s, Instructions.vnrcp)

	@JvmStatic
	fun vnsin(s: CpuState): Unit = unimplemented(s, Instructions.vnsin)

	@JvmStatic
	fun vrexp2(s: CpuState): Unit = unimplemented(s, Instructions.vrexp2)

	@JvmStatic
	fun vsat0(s: CpuState): Unit = unimplemented(s, Instructions.vsat0)

	@JvmStatic
	fun vsat1(s: CpuState): Unit = unimplemented(s, Instructions.vsat1)

	@JvmStatic
	fun vcst(s: CpuState): Unit = unimplemented(s, Instructions.vcst)

	@JvmStatic
	fun vmmul(s: CpuState): Unit = unimplemented(s, Instructions.vmmul)

	@JvmStatic
	fun vhdp(s: CpuState): Unit = unimplemented(s, Instructions.vhdp)

	@JvmStatic
	fun vcrs_t(s: CpuState): Unit = unimplemented(s, Instructions.vcrs_t)

	@JvmStatic
	fun vcrsp_t(s: CpuState): Unit = unimplemented(s, Instructions.vcrsp_t)

	@JvmStatic
	fun vi2c(s: CpuState): Unit = unimplemented(s, Instructions.vi2c)

	@JvmStatic
	fun vi2uc(s: CpuState): Unit = unimplemented(s, Instructions.vi2uc)

	@JvmStatic
	fun vtfm2(s: CpuState): Unit = unimplemented(s, Instructions.vtfm2)

	@JvmStatic
	fun vtfm3(s: CpuState): Unit = unimplemented(s, Instructions.vtfm3)

	@JvmStatic
	fun vtfm4(s: CpuState): Unit = unimplemented(s, Instructions.vtfm4)

	@JvmStatic
	fun vhtfm2(s: CpuState): Unit = unimplemented(s, Instructions.vhtfm2)

	@JvmStatic
	fun vhtfm3(s: CpuState): Unit = unimplemented(s, Instructions.vhtfm3)

	@JvmStatic
	fun vhtfm4(s: CpuState): Unit = unimplemented(s, Instructions.vhtfm4)

	@JvmStatic
	fun vsrt3(s: CpuState): Unit = unimplemented(s, Instructions.vsrt3)

	@JvmStatic
	fun vfad(s: CpuState): Unit = unimplemented(s, Instructions.vfad)

	@JvmStatic
	fun vmin(s: CpuState): Unit = unimplemented(s, Instructions.vmin)

	@JvmStatic
	fun vmax(s: CpuState): Unit = unimplemented(s, Instructions.vmax)

	@JvmStatic
	fun vadd(s: CpuState): Unit = unimplemented(s, Instructions.vadd)

	@JvmStatic
	fun vsub(s: CpuState): Unit = unimplemented(s, Instructions.vsub)

	@JvmStatic
	fun vdiv(s: CpuState): Unit = unimplemented(s, Instructions.vdiv)

	@JvmStatic
	fun vmul(s: CpuState): Unit = unimplemented(s, Instructions.vmul)

	@JvmStatic
	fun vidt(s: CpuState): Unit = unimplemented(s, Instructions.vidt)

	@JvmStatic
	fun vmidt(s: CpuState): Unit = unimplemented(s, Instructions.vmidt)

	@JvmStatic
	fun viim(s: CpuState): Unit = unimplemented(s, Instructions.viim)

	@JvmStatic
	fun vmmov(s: CpuState): Unit = unimplemented(s, Instructions.vmmov)

	@JvmStatic
	fun vmzero(s: CpuState): Unit = unimplemented(s, Instructions.vmzero)

	@JvmStatic
	fun vmone(s: CpuState): Unit = unimplemented(s, Instructions.vmone)

	@JvmStatic
	fun vnop(s: CpuState): Unit = unimplemented(s, Instructions.vnop)

	@JvmStatic
	fun vsync(s: CpuState): Unit = unimplemented(s, Instructions.vsync)

	@JvmStatic
	fun vflush(s: CpuState): Unit = unimplemented(s, Instructions.vflush)

	@JvmStatic
	fun vpfxd(s: CpuState): Unit = unimplemented(s, Instructions.vpfxd)

	@JvmStatic
	fun vpfxs(s: CpuState): Unit = unimplemented(s, Instructions.vpfxs)

	@JvmStatic
	fun vpfxt(s: CpuState): Unit = unimplemented(s, Instructions.vpfxt)

	@JvmStatic
	fun vdet(s: CpuState): Unit = unimplemented(s, Instructions.vdet)

	@JvmStatic
	fun vrnds(s: CpuState): Unit = unimplemented(s, Instructions.vrnds)

	@JvmStatic
	fun vrndi(s: CpuState): Unit = unimplemented(s, Instructions.vrndi)

	@JvmStatic
	fun vrndf1(s: CpuState): Unit = unimplemented(s, Instructions.vrndf1)

	@JvmStatic
	fun vrndf2(s: CpuState): Unit = unimplemented(s, Instructions.vrndf2)

	@JvmStatic
	fun vcmp(s: CpuState): Unit = unimplemented(s, Instructions.vcmp)

	@JvmStatic
	fun vcmovf(s: CpuState): Unit = unimplemented(s, Instructions.vcmovf)

	@JvmStatic
	fun vcmovt(s: CpuState): Unit = unimplemented(s, Instructions.vcmovt)

	@JvmStatic
	fun vavg(s: CpuState): Unit = unimplemented(s, Instructions.vavg)

	@JvmStatic
	fun vf2id(s: CpuState): Unit = unimplemented(s, Instructions.vf2id)

	@JvmStatic
	fun vf2in(s: CpuState): Unit = unimplemented(s, Instructions.vf2in)

	@JvmStatic
	fun vf2iu(s: CpuState): Unit = unimplemented(s, Instructions.vf2iu)

	@JvmStatic
	fun vf2iz(s: CpuState): Unit = unimplemented(s, Instructions.vf2iz)

	@JvmStatic
	fun vi2f(s: CpuState): Unit = unimplemented(s, Instructions.vi2f)

	@JvmStatic
	fun vscmp(s: CpuState): Unit = unimplemented(s, Instructions.vscmp)

	@JvmStatic
	fun vmscl(s: CpuState): Unit = unimplemented(s, Instructions.vmscl)

	@JvmStatic
	fun vt4444_q(s: CpuState): Unit = unimplemented(s, Instructions.vt4444_q)

	@JvmStatic
	fun vt5551_q(s: CpuState): Unit = unimplemented(s, Instructions.vt5551_q)

	@JvmStatic
	fun vt5650_q(s: CpuState): Unit = unimplemented(s, Instructions.vt5650_q)

	@JvmStatic
	fun vmfvc(s: CpuState): Unit = unimplemented(s, Instructions.vmfvc)

	@JvmStatic
	fun vmtvc(s: CpuState): Unit = unimplemented(s, Instructions.vmtvc)

	@JvmStatic
	fun mfvme(s: CpuState): Unit = unimplemented(s, Instructions.mfvme)

	@JvmStatic
	fun mtvme(s: CpuState): Unit = unimplemented(s, Instructions.mtvme)

	@JvmStatic
	fun sv_s(s: CpuState): Unit = unimplemented(s, Instructions.sv_s)

	@JvmStatic
	fun vfim(s: CpuState): Unit = unimplemented(s, Instructions.vfim)

	@JvmStatic
	fun svl_q(s: CpuState): Unit = unimplemented(s, Instructions.svl_q)

	@JvmStatic
	fun svr_q(s: CpuState): Unit = unimplemented(s, Instructions.svr_q)

	@JvmStatic
	fun vbfy1(s: CpuState): Unit = unimplemented(s, Instructions.vbfy1)

	@JvmStatic
	fun vbfy2(s: CpuState): Unit = unimplemented(s, Instructions.vbfy2)

	@JvmStatic
	fun vf2h(s: CpuState): Unit = unimplemented(s, Instructions.vf2h)

	@JvmStatic
	fun vh2f(s: CpuState): Unit = unimplemented(s, Instructions.vh2f)

	@JvmStatic
	fun vi2s(s: CpuState): Unit = unimplemented(s, Instructions.vi2s)

	@JvmStatic
	fun vi2us(s: CpuState): Unit = unimplemented(s, Instructions.vi2us)

	@JvmStatic
	fun vlgb(s: CpuState): Unit = unimplemented(s, Instructions.vlgb)

	@JvmStatic
	fun vqmul(s: CpuState): Unit = unimplemented(s, Instructions.vqmul)

	@JvmStatic
	fun vs2i(s: CpuState): Unit = unimplemented(s, Instructions.vs2i)

	@JvmStatic
	fun vc2i(s: CpuState): Unit = unimplemented(s, Instructions.vc2i)

	@JvmStatic
	fun vuc2i(s: CpuState): Unit = unimplemented(s, Instructions.vuc2i)

	@JvmStatic
	fun vsbn(s: CpuState): Unit = unimplemented(s, Instructions.vsbn)

	@JvmStatic
	fun vsbz(s: CpuState): Unit = unimplemented(s, Instructions.vsbz)

	@JvmStatic
	fun vsocp(s: CpuState): Unit = unimplemented(s, Instructions.vsocp)

	@JvmStatic
	fun vsrt1(s: CpuState): Unit = unimplemented(s, Instructions.vsrt1)

	@JvmStatic
	fun vsrt2(s: CpuState): Unit = unimplemented(s, Instructions.vsrt2)

	@JvmStatic
	fun vsrt4(s: CpuState): Unit = unimplemented(s, Instructions.vsrt4)

	@JvmStatic
	fun vus2i(s: CpuState): Unit = unimplemented(s, Instructions.vus2i)

	@JvmStatic
	fun vwbn(s: CpuState): Unit = unimplemented(s, Instructions.vwbn)

	@JvmStatic
	fun bvf(s: CpuState): Unit = unimplemented(s, Instructions.bvf)

	@JvmStatic
	fun bvt(s: CpuState): Unit = unimplemented(s, Instructions.bvt)

	@JvmStatic
	fun bvfl(s: CpuState): Unit = unimplemented(s, Instructions.bvfl)

	@JvmStatic
	fun bvtl(s: CpuState): Unit = unimplemented(s, Instructions.bvtl)
}
*/
