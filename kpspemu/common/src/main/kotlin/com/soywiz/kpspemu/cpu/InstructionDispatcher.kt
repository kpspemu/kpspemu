package com.soywiz.kpspemu.cpu

import com.soywiz.korio.lang.format

class InstructionDispatcher<T>(val e: InstructionEvaluator<T>) {
	fun dispatch(s: T, pc: Int, i: Int): Unit {
		when ((i and 0xFC000000.toInt())) {
			0x00000000.toInt() ->
				when ((i and 0x0000003F.toInt())) {
					0x00000020.toInt() ->
						return e.add(s)
					0x00000021.toInt() ->
						return e.addu(s)
					0x00000022.toInt() ->
						return e.sub(s)
					0x00000023.toInt() ->
						return e.subu(s)
					0x00000024.toInt() ->
						return e.and(s)
					0x00000027.toInt() ->
						return e.nor(s)
					0x00000025.toInt() ->
						return e.or(s)
					0x00000026.toInt() ->
						return e.xor(s)
					0x00000000.toInt() ->
						return e.sll(s)
					0x00000004.toInt() ->
						return e.sllv(s)
					0x00000003.toInt() ->
						return e.sra(s)
					0x00000007.toInt() ->
						return e.srav(s)
					0x00000002.toInt() ->
						when ((i and 0xFFE00000.toInt())) {
							0x00000000.toInt() ->
								return e.srl(s)
							0x00200000.toInt() ->
								return e.rotr(s)
							else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (0) failed mask 0x%08X".format(i, pc, -2097152))
						}
					0x00000006.toInt() ->
						when ((i and 0xFC0007C0.toInt())) {
							0x00000000.toInt() ->
								return e.srlv(s)
							0x00000040.toInt() ->
								return e.rotrv(s)
							else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (1) failed mask 0x%08X".format(i, pc, -67106880))
						}
					0x0000002A.toInt() ->
						return e.slt(s)
					0x0000002B.toInt() ->
						return e.sltu(s)
					0x0000002C.toInt() ->
						return e.max(s)
					0x0000002D.toInt() ->
						return e.min(s)
					0x0000001A.toInt() ->
						return e.div(s)
					0x0000001B.toInt() ->
						return e.divu(s)
					0x00000018.toInt() ->
						return e.mult(s)
					0x00000019.toInt() ->
						return e.multu(s)
					0x0000001C.toInt() ->
						return e.madd(s)
					0x0000001D.toInt() ->
						return e.maddu(s)
					0x0000002E.toInt() ->
						return e.msub(s)
					0x0000002F.toInt() ->
						return e.msubu(s)
					0x00000010.toInt() ->
						return e.mfhi(s)
					0x00000012.toInt() ->
						return e.mflo(s)
					0x00000011.toInt() ->
						return e.mthi(s)
					0x00000013.toInt() ->
						return e.mtlo(s)
					0x0000000A.toInt() ->
						return e.movz(s)
					0x0000000B.toInt() ->
						return e.movn(s)
					0x00000016.toInt() ->
						return e.clz(s)
					0x00000017.toInt() ->
						return e.clo(s)
					0x00000008.toInt() ->
						return e.jr(s)
					0x00000009.toInt() ->
						return e.jalr(s)
					0x0000000C.toInt() ->
						return e.syscall(s)
					0x0000000F.toInt() ->
						return e.sync(s)
					0x0000000D.toInt() ->
						return e._break(s)
					else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (2) failed mask 0x%08X".format(i, pc, 63))
				}
			0x20000000.toInt() ->
				return e.addi(s)
			0x24000000.toInt() ->
				return e.addiu(s)
			0x30000000.toInt() ->
				return e.andi(s)
			0x34000000.toInt() ->
				return e.ori(s)
			0x38000000.toInt() ->
				return e.xori(s)
			0x28000000.toInt() ->
				return e.slti(s)
			0x2C000000.toInt() ->
				return e.sltiu(s)
			0x3C000000.toInt() ->
				return e.lui(s)
			0x7C000000.toInt() ->
				when ((i and 0x0000003F.toInt())) {
					0x00000020.toInt() ->
						when ((i and 0xFFE007C0.toInt())) {
							0x7C000400.toInt() ->
								return e.seb(s)
							0x7C000600.toInt() ->
								return e.seh(s)
							0x7C000500.toInt() ->
								return e.bitrev(s)
							0x7C000080.toInt() ->
								return e.wsbh(s)
							0x7C0000C0.toInt() ->
								return e.wsbw(s)
							else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (3) failed mask 0x%08X".format(i, pc, -2095168))
						}
					0x00000000.toInt() ->
						return e.ext(s)
					0x00000004.toInt() ->
						return e.ins(s)
					else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (4) failed mask 0x%08X".format(i, pc, 63))
				}
			0x10000000.toInt() ->
				return e.beq(s)
			0x50000000.toInt() ->
				return e.beql(s)
			0x04000000.toInt() ->
				when ((i and 0x001F0000.toInt())) {
					0x00010000.toInt() ->
						return e.bgez(s)
					0x00030000.toInt() ->
						return e.bgezl(s)
					0x00110000.toInt() ->
						return e.bgezal(s)
					0x00130000.toInt() ->
						return e.bgezall(s)
					0x00000000.toInt() ->
						return e.bltz(s)
					0x00020000.toInt() ->
						return e.bltzl(s)
					0x00100000.toInt() ->
						return e.bltzal(s)
					0x00120000.toInt() ->
						return e.bltzall(s)
					else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (5) failed mask 0x%08X".format(i, pc, 2031616))
				}
			0x18000000.toInt() ->
				return e.blez(s)
			0x58000000.toInt() ->
				return e.blezl(s)
			0x1C000000.toInt() ->
				return e.bgtz(s)
			0x5C000000.toInt() ->
				return e.bgtzl(s)
			0x14000000.toInt() ->
				return e.bne(s)
			0x54000000.toInt() ->
				return e.bnel(s)
			0x08000000.toInt() ->
				return e.j(s)
			0x0C000000.toInt() ->
				return e.jal(s)
			0x44000000.toInt() ->
				when ((i and 0x03E00000.toInt())) {
					0x01000000.toInt() ->
						when ((i and 0xFC1F0000.toInt())) {
							0x44000000.toInt() ->
								return e.bc1f(s)
							0x44010000.toInt() ->
								return e.bc1t(s)
							0x44020000.toInt() ->
								return e.bc1fl(s)
							0x44030000.toInt() ->
								return e.bc1tl(s)
							else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (6) failed mask 0x%08X".format(i, pc, -65077248))
						}
					0x02000000.toInt() ->
						when ((i and 0xFC00003F.toInt())) {
							0x44000000.toInt() ->
								return e.add_s(s)
							0x44000001.toInt() ->
								return e.sub_s(s)
							0x44000002.toInt() ->
								return e.mul_s(s)
							0x44000003.toInt() ->
								return e.div_s(s)
							0x44000004.toInt() ->
								return e.sqrt_s(s)
							0x44000005.toInt() ->
								return e.abs_s(s)
							0x44000006.toInt() ->
								return e.mov_s(s)
							0x44000007.toInt() ->
								return e.neg_s(s)
							0x4400000C.toInt() ->
								return e.round_w_s(s)
							0x4400000D.toInt() ->
								return e.trunc_w_s(s)
							0x4400000E.toInt() ->
								return e.ceil_w_s(s)
							0x4400000F.toInt() ->
								return e.floor_w_s(s)
							0x44000024.toInt() ->
								return e.cvt_w_s(s)
							0x44000030.toInt() ->
								return e.c_f_s(s)
							0x44000031.toInt() ->
								return e.c_un_s(s)
							0x44000032.toInt() ->
								return e.c_eq_s(s)
							0x44000033.toInt() ->
								return e.c_ueq_s(s)
							0x44000034.toInt() ->
								return e.c_olt_s(s)
							0x44000035.toInt() ->
								return e.c_ult_s(s)
							0x44000036.toInt() ->
								return e.c_ole_s(s)
							0x44000037.toInt() ->
								return e.c_ule_s(s)
							0x44000038.toInt() ->
								return e.c_sf_s(s)
							0x44000039.toInt() ->
								return e.c_ngle_s(s)
							0x4400003A.toInt() ->
								return e.c_seq_s(s)
							0x4400003B.toInt() ->
								return e.c_ngl_s(s)
							0x4400003C.toInt() ->
								return e.c_lt_s(s)
							0x4400003D.toInt() ->
								return e.c_nge_s(s)
							0x4400003E.toInt() ->
								return e.c_le_s(s)
							0x4400003F.toInt() ->
								return e.c_ngt_s(s)
							else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (7) failed mask 0x%08X".format(i, pc, -67108801))
						}
					0x02800000.toInt() ->
						return e.cvt_s_w(s)
					0x00000000.toInt() ->
						return e.mfc1(s)
					0x00800000.toInt() ->
						return e.mtc1(s)
					0x00400000.toInt() ->
						return e.cfc1(s)
					0x00C00000.toInt() ->
						return e.ctc1(s)
					else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (8) failed mask 0x%08X".format(i, pc, 65011712))
				}
			0x80000000.toInt() ->
				return e.lb(s)
			0x84000000.toInt() ->
				return e.lh(s)
			0x8C000000.toInt() ->
				return e.lw(s)
			0x88000000.toInt() ->
				return e.lwl(s)
			0x98000000.toInt() ->
				return e.lwr(s)
			0x90000000.toInt() ->
				return e.lbu(s)
			0x94000000.toInt() ->
				return e.lhu(s)
			0xA0000000.toInt() ->
				return e.sb(s)
			0xA4000000.toInt() ->
				return e.sh(s)
			0xAC000000.toInt() ->
				return e.sw(s)
			0xA8000000.toInt() ->
				return e.swl(s)
			0xB8000000.toInt() ->
				return e.swr(s)
			0xC0000000.toInt() ->
				return e.ll(s)
			0xE0000000.toInt() ->
				return e.sc(s)
			0xC4000000.toInt() ->
				return e.lwc1(s)
			0xE4000000.toInt() ->
				return e.swc1(s)
			0xBC000000.toInt() ->
				return e.cache(s)
			0x70000000.toInt() ->
				when ((i and 0x000007FF.toInt())) {
					0x0000003F.toInt() ->
						return e.dbreak(s)
					0x00000000.toInt() ->
						return e.halt(s)
					0x0000003E.toInt() ->
						return e.dret(s)
					0x00000024.toInt() ->
						return e.mfic(s)
					0x00000026.toInt() ->
						return e.mtic(s)
					0x0000003D.toInt() ->
						when ((i and 0xFFE00000.toInt())) {
							0x70000000.toInt() ->
								return e.mfdr(s)
							0x70800000.toInt() ->
								return e.mtdr(s)
							else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (9) failed mask 0x%08X".format(i, pc, -2097152))
						}
					else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (10) failed mask 0x%08X".format(i, pc, 2047))
				}
			0x40000000.toInt() ->
				when ((i and 0x03E007FF.toInt())) {
					0x02000018.toInt() ->
						return e.eret(s)
					0x00400000.toInt() ->
						return e.cfc0(s)
					0x00C00000.toInt() ->
						return e.ctc0(s)
					0x00000000.toInt() ->
						return e.mfc0(s)
					0x00800000.toInt() ->
						return e.mtc0(s)
					else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (11) failed mask 0x%08X".format(i, pc, 65013759))
				}
			0x48000000.toInt() ->
				when ((i and 0x03E00000.toInt())) {
					0x00600000.toInt() ->
						when ((i and 0xFC00FF80.toInt())) {
							0x48000000.toInt() ->
								return e.mfv(s)
							0x48000080.toInt() ->
								return e.mfvc(s)
							else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (12) failed mask 0x%08X".format(i, pc, -67043456))
						}
					0x00E00000.toInt() ->
						when ((i and 0xFC00FF80.toInt())) {
							0x48000000.toInt() ->
								return e.mtv(s)
							0x48000080.toInt() ->
								return e.mtvc(s)
							else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (13) failed mask 0x%08X".format(i, pc, -67043456))
						}
					0x01000000.toInt() ->
						when ((i and 0xFC030000.toInt())) {
							0x48000000.toInt() ->
								return e.bvf(s)
							0x48010000.toInt() ->
								return e.bvt(s)
							0x48020000.toInt() ->
								return e.bvfl(s)
							0x48030000.toInt() ->
								return e.bvtl(s)
							else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (14) failed mask 0x%08X".format(i, pc, -66912256))
						}
					else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (15) failed mask 0x%08X".format(i, pc, 65011712))
				}
			0xC8000000.toInt() ->
				return e.lv_s(s)
			0xD8000000.toInt() ->
				return e.lv_q(s)
			0xD4000000.toInt() ->
				when ((i and 0x00000002.toInt())) {
					0x00000000.toInt() ->
						return e.lvl_q(s)
					0x00000002.toInt() ->
						return e.lvr_q(s)
					else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (16) failed mask 0x%08X".format(i, pc, 2))
				}
			0xF8000000.toInt() ->
				return e.sv_q(s)
			0x64000000.toInt() ->
				when ((i and 0x03800000.toInt())) {
					0x00800000.toInt() ->
						return e.vdot(s)
					0x01000000.toInt() ->
						return e.vscl(s)
					0x02000000.toInt() ->
						return e.vhdp(s)
					0x02800000.toInt() ->
						return e.vcrs_t(s)
					0x00000000.toInt() ->
						return e.vmul(s)
					0x03000000.toInt() ->
						return e.vdet(s)
					else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (17) failed mask 0x%08X".format(i, pc, 58720256))
				}
			0x6C000000.toInt() ->
				when ((i and 0x03800000.toInt())) {
					0x03000000.toInt() ->
						return e.vsge(s)
					0x03800000.toInt() ->
						return e.vslt(s)
					0x01000000.toInt() ->
						return e.vmin(s)
					0x01800000.toInt() ->
						return e.vmax(s)
					0x00000000.toInt() ->
						return e.vcmp(s)
					0x02800000.toInt() ->
						return e.vscmp(s)
					else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (18) failed mask 0x%08X".format(i, pc, 58720256))
				}
			0xF0000000.toInt() ->
				when ((i and 0x03800000.toInt())) {
					0x03800000.toInt() ->
						when ((i and 0xFC600000.toInt())) {
							0xF0200000.toInt() ->
								return e.vrot(s)
							0xF0000000.toInt() ->
								when ((i and 0x039F0000.toInt())) {
									0x03830000.toInt() ->
										return e.vmidt(s)
									0x03800000.toInt() ->
										return e.vmmov(s)
									0x03860000.toInt() ->
										return e.vmzero(s)
									0x03870000.toInt() ->
										return e.vmone(s)
									else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (19) failed mask 0x%08X".format(i, pc, 60751872))
								}
							else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (20) failed mask 0x%08X".format(i, pc, -60817408))
						}
					0x00000000.toInt() ->
						return e.vmmul(s)
					0x02800000.toInt() ->
						when ((i and 0xFC008080.toInt())) {
							0xF0008000.toInt() ->
								return e.vcrsp_t(s)
							0xF0008080.toInt() ->
								return e.vqmul(s)
							else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (21) failed mask 0x%08X".format(i, pc, -67075968))
						}
					0x00800000.toInt() ->
						when ((i and 0xFC008080.toInt())) {
							0xF0000080.toInt() ->
								return e.vtfm2(s)
							0xF0000000.toInt() ->
								return e.vhtfm2(s)
							else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (22) failed mask 0x%08X".format(i, pc, -67075968))
						}
					0x01000000.toInt() ->
						when ((i and 0xFC008080.toInt())) {
							0xF0008000.toInt() ->
								return e.vtfm3(s)
							0xF0000080.toInt() ->
								return e.vhtfm3(s)
							else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (23) failed mask 0x%08X".format(i, pc, -67075968))
						}
					0x01800000.toInt() ->
						when ((i and 0xFC008080.toInt())) {
							0xF0008080.toInt() ->
								return e.vtfm4(s)
							0xF0008000.toInt() ->
								return e.vhtfm4(s)
							else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (24) failed mask 0x%08X".format(i, pc, -67075968))
						}
					0x02000000.toInt() ->
						return e.vmscl(s)
					else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (25) failed mask 0x%08X".format(i, pc, 58720256))
				}
			0xD0000000.toInt() ->
				when ((i and 0x03000000.toInt())) {
					0x00000000.toInt() ->
						when ((i and 0xFCE00000.toInt())) {
							0xD0000000.toInt() ->
								when ((i and 0x031F0000.toInt())) {
									0x00060000.toInt() ->
										return e.vzero(s)
									0x00070000.toInt() ->
										return e.vone(s)
									0x00000000.toInt() ->
										return e.vmov(s)
									0x00010000.toInt() ->
										return e.vabs(s)
									0x00020000.toInt() ->
										return e.vneg(s)
									0x00100000.toInt() ->
										return e.vrcp(s)
									0x00110000.toInt() ->
										return e.vrsq(s)
									0x00120000.toInt() ->
										return e.vsin(s)
									0x00130000.toInt() ->
										return e.vcos(s)
									0x00140000.toInt() ->
										return e.vexp2(s)
									0x00150000.toInt() ->
										return e.vlog2(s)
									0x00160000.toInt() ->
										return e.vsqrt(s)
									0x00170000.toInt() ->
										return e.vasin(s)
									0x00180000.toInt() ->
										return e.vnrcp(s)
									0x001A0000.toInt() ->
										return e.vnsin(s)
									0x001C0000.toInt() ->
										return e.vrexp2(s)
									0x00040000.toInt() ->
										return e.vsat0(s)
									0x00050000.toInt() ->
										return e.vsat1(s)
									0x00030000.toInt() ->
										return e.vidt(s)
									else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (26) failed mask 0x%08X".format(i, pc, 52363264))
								}
							0xD0400000.toInt() ->
								when ((i and 0x031F0000.toInt())) {
									0x00040000.toInt() ->
										return e.vocp(s)
									0x000A0000.toInt() ->
										return e.vsgn(s)
									0x00080000.toInt() ->
										return e.vsrt3(s)
									0x00060000.toInt() ->
										return e.vfad(s)
									0x00070000.toInt() ->
										return e.vavg(s)
									0x00190000.toInt() ->
										return e.vt4444_q(s)
									0x001A0000.toInt() ->
										return e.vt5551_q(s)
									0x001B0000.toInt() ->
										return e.vt5650_q(s)
									0x00100000.toInt() ->
										return e.vmfvc(s)
									0x00110000.toInt() ->
										return e.vmtvc(s)
									0x00020000.toInt() ->
										return e.vbfy1(s)
									0x00030000.toInt() ->
										return e.vbfy2(s)
									0x00050000.toInt() ->
										return e.vsocp(s)
									0x00000000.toInt() ->
										return e.vsrt1(s)
									0x00010000.toInt() ->
										return e.vsrt2(s)
									0x00090000.toInt() ->
										return e.vsrt4(s)
									else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (27) failed mask 0x%08X".format(i, pc, 52363264))
								}
							0xD0600000.toInt() ->
								return e.vcst(s)
							0xD0200000.toInt() ->
								when ((i and 0x031F0000.toInt())) {
									0x001D0000.toInt() ->
										return e.vi2c(s)
									0x001C0000.toInt() ->
										return e.vi2uc(s)
									0x00000000.toInt() ->
										return e.vrnds(s)
									0x00010000.toInt() ->
										return e.vrndi(s)
									0x00020000.toInt() ->
										return e.vrndf1(s)
									0x00030000.toInt() ->
										return e.vrndf2(s)
									0x00120000.toInt() ->
										return e.vf2h(s)
									0x00130000.toInt() ->
										return e.vh2f(s)
									0x001F0000.toInt() ->
										return e.vi2s(s)
									0x001E0000.toInt() ->
										return e.vi2us(s)
									0x00170000.toInt() ->
										return e.vlgb(s)
									0x001B0000.toInt() ->
										return e.vs2i(s)
									0x00190000.toInt() ->
										return e.vc2i(s)
									0x00180000.toInt() ->
										return e.vuc2i(s)
									0x00160000.toInt() ->
										return e.vsbz(s)
									0x001A0000.toInt() ->
										return e.vus2i(s)
									else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (28) failed mask 0x%08X".format(i, pc, 52363264))
								}
							else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (29) failed mask 0x%08X".format(i, pc, -52428800))
						}
					0x02000000.toInt() ->
						when ((i and 0xFCE00000.toInt())) {
							0xD0A00000.toInt() ->
								when ((i and 0x03180000.toInt())) {
									0x02080000.toInt() ->
										return e.vcmovf(s)
									0x02000000.toInt() ->
										return e.vcmovt(s)
									else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (30) failed mask 0x%08X".format(i, pc, 51904512))
								}
							0xD0600000.toInt() ->
								return e.vf2id(s)
							0xD0000000.toInt() ->
								return e.vf2in(s)
							0xD0400000.toInt() ->
								return e.vf2iu(s)
							0xD0200000.toInt() ->
								return e.vf2iz(s)
							0xD0800000.toInt() ->
								return e.vi2f(s)
							else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (31) failed mask 0x%08X".format(i, pc, -52428800))
						}
					0x03000000.toInt() ->
						return e.vwbn(s)
					else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (32) failed mask 0x%08X".format(i, pc, 50331648))
				}
			0x60000000.toInt() ->
				when ((i and 0x03800000.toInt())) {
					0x00000000.toInt() ->
						return e.vadd(s)
					0x00800000.toInt() ->
						return e.vsub(s)
					0x03800000.toInt() ->
						return e.vdiv(s)
					0x01000000.toInt() ->
						return e.vsbn(s)
					else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (33) failed mask 0x%08X".format(i, pc, 58720256))
				}
			0xDC000000.toInt() ->
				when ((i and 0x03000000.toInt())) {
					0x03000000.toInt() ->
						when ((i and 0xFC800000.toInt())) {
							0xDC000000.toInt() ->
								return e.viim(s)
							0xDC800000.toInt() ->
								return e.vfim(s)
							else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (34) failed mask 0x%08X".format(i, pc, -58720256))
						}
					0x02000000.toInt() ->
						return e.vpfxd(s)
					0x00000000.toInt() ->
						return e.vpfxs(s)
					0x01000000.toInt() ->
						return e.vpfxt(s)
					else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (35) failed mask 0x%08X".format(i, pc, 50331648))
				}
			0xFC000000.toInt() ->
				when ((i and 0x03FFFFFF.toInt())) {
					0x03FF0000.toInt() ->
						return e.vnop(s)
					0x03FF0320.toInt() ->
						return e.vsync(s)
					0x03FF040D.toInt() ->
						return e.vflush(s)
					else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (36) failed mask 0x%08X".format(i, pc, 67108863))
				}
			0x68000000.toInt() ->
				return e.mfvme(s)
			0xB0000000.toInt() ->
				return e.mtvme(s)
			0xE8000000.toInt() ->
				return e.sv_s(s)
			0xF4000000.toInt() ->
				when ((i and 0x00000002.toInt())) {
					0x00000000.toInt() ->
						return e.svl_q(s)
					0x00000002.toInt() ->
						return e.svr_q(s)
					else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (37) failed mask 0x%08X".format(i, pc, 2))
				}
			else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (38) failed mask 0x%08X".format(i, pc, -67108864))
		}
	}
}

fun InstructionDispatcher<CpuState>.dispatch(s: CpuState) = this.dispatch(s, s.PC, s.I)