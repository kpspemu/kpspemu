package com.soywiz.kpspemu.cpu

import com.soywiz.korio.lang.format

@Suppress("RemoveRedundantCallsOfConversionMethods", "LiftReturnOrAssignment", "RedundantUnitReturnType")
class InstructionDispatcher<T>(val e: InstructionEvaluator<T>) {
	fun dispatch(s: T, pc: Int, i: Int): Unit {
		when (((i ushr 26) and 63)) {
			0 ->
				when (((i ushr 0) and 63)) {
					0 -> return e.sll(s)
					2 ->
						when (((i ushr 21) and 2047)) {
							0 -> return e.srl(s)
							1 -> return e.rotr(s)
							else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (0) failed mask 0x%08X".format(i, pc, -2097152))
						}
					3 -> return e.sra(s)
					4 -> return e.sllv(s)
					6 ->
						when (((i ushr 6) and 66060319)) {
							0 -> return e.srlv(s)
							1 -> return e.rotrv(s)
							else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (1) failed mask 0x%08X".format(i, pc, -67106880))
						}
					7 -> return e.srav(s)
					8 -> return e.jr(s)
					9 -> return e.jalr(s)
					10 -> return e.movz(s)
					11 -> return e.movn(s)
					12 -> return e.syscall(s)
					13 -> return e._break(s)
					15 -> return e.sync(s)
					16 -> return e.mfhi(s)
					17 -> return e.mthi(s)
					18 -> return e.mflo(s)
					19 -> return e.mtlo(s)
					22 -> return e.clz(s)
					23 -> return e.clo(s)
					24 -> return e.mult(s)
					25 -> return e.multu(s)
					26 -> return e.div(s)
					27 -> return e.divu(s)
					28 -> return e.madd(s)
					29 -> return e.maddu(s)
					32 -> return e.add(s)
					33 -> return e.addu(s)
					34 -> return e.sub(s)
					35 -> return e.subu(s)
					36 -> return e.and(s)
					37 -> return e.or(s)
					38 -> return e.xor(s)
					39 -> return e.nor(s)
					42 -> return e.slt(s)
					43 -> return e.sltu(s)
					44 -> return e.max(s)
					45 -> return e.min(s)
					46 -> return e.msub(s)
					47 -> return e.msubu(s)
					else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (2) failed mask 0x%08X".format(i, pc, 63))
				}
			1 ->
				when (((i ushr 16) and 31)) {
					0 -> return e.bltz(s)
					1 -> return e.bgez(s)
					2 -> return e.bltzl(s)
					3 -> return e.bgezl(s)
					16 -> return e.bltzal(s)
					17 -> return e.bgezal(s)
					18 -> return e.bltzall(s)
					19 -> return e.bgezall(s)
					else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (3) failed mask 0x%08X".format(i, pc, 2031616))
				}
			2 -> return e.j(s)
			3 -> return e.jal(s)
			4 -> return e.beq(s)
			5 -> return e.bne(s)
			6 -> return e.blez(s)
			7 -> return e.bgtz(s)
			8 -> return e.addi(s)
			9 -> return e.addiu(s)
			10 -> return e.slti(s)
			11 -> return e.sltiu(s)
			12 -> return e.andi(s)
			13 -> return e.ori(s)
			14 -> return e.xori(s)
			15 -> return e.lui(s)
			16 ->
				when (((i ushr 0) and 65013759)) {
					0 -> return e.mfc0(s)
					4194304 -> return e.cfc0(s)
					8388608 -> return e.mtc0(s)
					12582912 -> return e.ctc0(s)
					33554456 -> return e.eret(s)
					else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (4) failed mask 0x%08X".format(i, pc, 65013759))
				}
			17 ->
				when (((i ushr 21) and 31)) {
					0 -> return e.mfc1(s)
					2 -> return e.cfc1(s)
					4 -> return e.mtc1(s)
					6 -> return e.ctc1(s)
					8 ->
						when (((i ushr 16) and 64543)) {
							17408 -> return e.bc1f(s)
							17409 -> return e.bc1t(s)
							17410 -> return e.bc1fl(s)
							17411 -> return e.bc1tl(s)
							else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (5) failed mask 0x%08X".format(i, pc, -65077248))
						}
					16 ->
						when (((i ushr 0) and -67108801)) {
							1140850688 -> return e.add_s(s)
							1140850689 -> return e.sub_s(s)
							1140850690 -> return e.mul_s(s)
							1140850691 -> return e.div_s(s)
							1140850692 -> return e.sqrt_s(s)
							1140850693 -> return e.abs_s(s)
							1140850694 -> return e.mov_s(s)
							1140850695 -> return e.neg_s(s)
							1140850700 -> return e.round_w_s(s)
							1140850701 -> return e.trunc_w_s(s)
							1140850702 -> return e.ceil_w_s(s)
							1140850703 -> return e.floor_w_s(s)
							1140850724 -> return e.cvt_w_s(s)
							1140850736 -> return e.c_f_s(s)
							1140850737 -> return e.c_un_s(s)
							1140850738 -> return e.c_eq_s(s)
							1140850739 -> return e.c_ueq_s(s)
							1140850740 -> return e.c_olt_s(s)
							1140850741 -> return e.c_ult_s(s)
							1140850742 -> return e.c_ole_s(s)
							1140850743 -> return e.c_ule_s(s)
							1140850744 -> return e.c_sf_s(s)
							1140850745 -> return e.c_ngle_s(s)
							1140850746 -> return e.c_seq_s(s)
							1140850747 -> return e.c_ngl_s(s)
							1140850748 -> return e.c_lt_s(s)
							1140850749 -> return e.c_nge_s(s)
							1140850750 -> return e.c_le_s(s)
							1140850751 -> return e.c_ngt_s(s)
							else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (6) failed mask 0x%08X".format(i, pc, -67108801))
						}
					20 -> return e.cvt_s_w(s)
					else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (7) failed mask 0x%08X".format(i, pc, 65011712))
				}
			18 ->
				when (((i ushr 21) and 31)) {
					3 ->
						when (((i ushr 7) and 33030655)) {
							9437184 -> return e.mfv(s)
							9437185 -> return e.mfvc(s)
							else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (8) failed mask 0x%08X".format(i, pc, -67043456))
						}
					7 ->
						when (((i ushr 7) and 33030655)) {
							9437184 -> return e.mtv(s)
							9437185 -> return e.mtvc(s)
							else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (9) failed mask 0x%08X".format(i, pc, -67043456))
						}
					8 ->
						when (((i ushr 16) and 64515)) {
							18432 -> return e.bvf(s)
							18433 -> return e.bvt(s)
							18434 -> return e.bvfl(s)
							18435 -> return e.bvtl(s)
							else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (10) failed mask 0x%08X".format(i, pc, -66912256))
						}
					else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (11) failed mask 0x%08X".format(i, pc, 65011712))
				}
			20 -> return e.beql(s)
			21 -> return e.bnel(s)
			22 -> return e.blezl(s)
			23 -> return e.bgtzl(s)
			24 ->
				when (((i ushr 23) and 7)) {
					0 -> return e.vadd(s)
					1 -> return e.vsub(s)
					2 -> return e.vsbn(s)
					7 -> return e.vdiv(s)
					else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (12) failed mask 0x%08X".format(i, pc, 58720256))
				}
			25 ->
				when (((i ushr 23) and 7)) {
					0 -> return e.vmul(s)
					1 -> return e.vdot(s)
					2 -> return e.vscl(s)
					4 -> return e.vhdp(s)
					5 -> return e.vcrs_t(s)
					6 -> return e.vdet(s)
					else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (13) failed mask 0x%08X".format(i, pc, 58720256))
				}
			26 -> return e.mfvme(s)
			27 ->
				when (((i ushr 23) and 7)) {
					0 -> return e.vcmp(s)
					2 -> return e.vmin(s)
					3 -> return e.vmax(s)
					5 -> return e.vscmp(s)
					6 -> return e.vsge(s)
					7 -> return e.vslt(s)
					else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (14) failed mask 0x%08X".format(i, pc, 58720256))
				}
			28 ->
				when (((i ushr 0) and 2047)) {
					0 -> return e.halt(s)
					36 -> return e.mfic(s)
					38 -> return e.mtic(s)
					61 ->
						when (((i ushr 21) and 2047)) {
							896 -> return e.mfdr(s)
							900 -> return e.mtdr(s)
							else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (15) failed mask 0x%08X".format(i, pc, -2097152))
						}
					62 -> return e.dret(s)
					63 -> return e.dbreak(s)
					else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (16) failed mask 0x%08X".format(i, pc, 2047))
				}
			31 ->
				when (((i ushr 0) and 63)) {
					0 -> return e.ext(s)
					4 -> return e.ins(s)
					32 ->
						when (((i ushr 6) and 67076127)) {
							32505858 -> return e.wsbh(s)
							32505859 -> return e.wsbw(s)
							32505872 -> return e.seb(s)
							32505876 -> return e.bitrev(s)
							32505880 -> return e.seh(s)
							else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (17) failed mask 0x%08X".format(i, pc, -2095168))
						}
					else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (18) failed mask 0x%08X".format(i, pc, 63))
				}
			32 -> return e.lb(s)
			33 -> return e.lh(s)
			34 -> return e.lwl(s)
			35 -> return e.lw(s)
			36 -> return e.lbu(s)
			37 -> return e.lhu(s)
			38 -> return e.lwr(s)
			40 -> return e.sb(s)
			41 -> return e.sh(s)
			42 -> return e.swl(s)
			43 -> return e.sw(s)
			44 -> return e.mtvme(s)
			46 -> return e.swr(s)
			47 -> return e.cache(s)
			48 -> return e.ll(s)
			49 -> return e.lwc1(s)
			50 -> return e.lv_s(s)
			52 ->
				when (((i ushr 24) and 3)) {
					0 ->
						when (((i ushr 21) and 2023)) {
							1664 ->
								when (((i ushr 16) and 799)) {
									0 -> return e.vmov(s)
									1 -> return e.vabs(s)
									2 -> return e.vneg(s)
									3 -> return e.vidt(s)
									4 -> return e.vsat0(s)
									5 -> return e.vsat1(s)
									6 -> return e.vzero(s)
									7 -> return e.vone(s)
									16 -> return e.vrcp(s)
									17 -> return e.vrsq(s)
									18 -> return e.vsin(s)
									19 -> return e.vcos(s)
									20 -> return e.vexp2(s)
									21 -> return e.vlog2(s)
									22 -> return e.vsqrt(s)
									23 -> return e.vasin(s)
									24 -> return e.vnrcp(s)
									26 -> return e.vnsin(s)
									28 -> return e.vrexp2(s)
									else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (19) failed mask 0x%08X".format(i, pc, 52363264))
								}
							1665 ->
								when (((i ushr 16) and 799)) {
									0 -> return e.vrnds(s)
									1 -> return e.vrndi(s)
									2 -> return e.vrndf1(s)
									3 -> return e.vrndf2(s)
									18 -> return e.vf2h(s)
									19 -> return e.vh2f(s)
									22 -> return e.vsbz(s)
									23 -> return e.vlgb(s)
									24 -> return e.vuc2i(s)
									25 -> return e.vc2i(s)
									26 -> return e.vus2i(s)
									27 -> return e.vs2i(s)
									28 -> return e.vi2uc(s)
									29 -> return e.vi2c(s)
									30 -> return e.vi2us(s)
									31 -> return e.vi2s(s)
									else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (20) failed mask 0x%08X".format(i, pc, 52363264))
								}
							1666 ->
								when (((i ushr 16) and 799)) {
									0 -> return e.vsrt1(s)
									1 -> return e.vsrt2(s)
									2 -> return e.vbfy1(s)
									3 -> return e.vbfy2(s)
									4 -> return e.vocp(s)
									5 -> return e.vsocp(s)
									6 -> return e.vfad(s)
									7 -> return e.vavg(s)
									8 -> return e.vsrt3(s)
									9 -> return e.vsrt4(s)
									10 -> return e.vsgn(s)
									16 -> return e.vmfvc(s)
									17 -> return e.vmtvc(s)
									25 -> return e.vt4444_q(s)
									26 -> return e.vt5551_q(s)
									27 -> return e.vt5650_q(s)
									else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (21) failed mask 0x%08X".format(i, pc, 52363264))
								}
							1667 -> return e.vcst(s)
							else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (22) failed mask 0x%08X".format(i, pc, -52428800))
						}
					2 ->
						when (((i ushr 21) and 2023)) {
							1664 -> return e.vf2in(s)
							1665 -> return e.vf2iz(s)
							1666 -> return e.vf2iu(s)
							1667 -> return e.vf2id(s)
							1668 -> return e.vi2f(s)
							1669 ->
								when (((i ushr 19) and 99)) {
									64 -> return e.vcmovt(s)
									65 -> return e.vcmovf(s)
									else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (23) failed mask 0x%08X".format(i, pc, 51904512))
								}
							else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (24) failed mask 0x%08X".format(i, pc, -52428800))
						}
					3 -> return e.vwbn(s)
					else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (25) failed mask 0x%08X".format(i, pc, 50331648))
				}
			53 ->
				when (((i ushr 1) and 1)) {
					0 -> return e.lvl_q(s)
					1 -> return e.lvr_q(s)
					else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (26) failed mask 0x%08X".format(i, pc, 2))
				}
			54 -> return e.lv_q(s)
			55 ->
				when (((i ushr 24) and 3)) {
					0 -> return e.vpfxs(s)
					1 -> return e.vpfxt(s)
					2 -> return e.vpfxd(s)
					3 ->
						when (((i ushr 23) and 505)) {
							440 -> return e.viim(s)
							441 -> return e.vfim(s)
							else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (27) failed mask 0x%08X".format(i, pc, -58720256))
						}
					else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (28) failed mask 0x%08X".format(i, pc, 50331648))
				}
			56 -> return e.sc(s)
			57 -> return e.swc1(s)
			58 -> return e.sv_s(s)
			60 ->
				when (((i ushr 23) and 7)) {
					0 -> return e.vmmul(s)
					1 ->
						when (((i ushr 7) and 33030401)) {
							31457280 -> return e.vhtfm2(s)
							31457281 -> return e.vtfm2(s)
							else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (29) failed mask 0x%08X".format(i, pc, -67075968))
						}
					2 ->
						when (((i ushr 7) and 33030401)) {
							31457281 -> return e.vhtfm3(s)
							31457536 -> return e.vtfm3(s)
							else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (30) failed mask 0x%08X".format(i, pc, -67075968))
						}
					3 ->
						when (((i ushr 7) and 33030401)) {
							31457536 -> return e.vhtfm4(s)
							31457537 -> return e.vtfm4(s)
							else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (31) failed mask 0x%08X".format(i, pc, -67075968))
						}
					4 -> return e.vmscl(s)
					5 ->
						when (((i ushr 7) and 33030401)) {
							31457536 -> return e.vcrsp_t(s)
							31457537 -> return e.vqmul(s)
							else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (32) failed mask 0x%08X".format(i, pc, -67075968))
						}
					7 ->
						when (((i ushr 21) and 2019)) {
							1920 ->
								when (((i ushr 16) and 927)) {
									896 -> return e.vmmov(s)
									899 -> return e.vmidt(s)
									902 -> return e.vmzero(s)
									903 -> return e.vmone(s)
									else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (33) failed mask 0x%08X".format(i, pc, 60751872))
								}
							1921 -> return e.vrot(s)
							else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (34) failed mask 0x%08X".format(i, pc, -60817408))
						}
					else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (35) failed mask 0x%08X".format(i, pc, 58720256))
				}
			61 ->
				when (((i ushr 1) and 1)) {
					0 -> return e.svl_q(s)
					1 -> return e.svr_q(s)
					else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (36) failed mask 0x%08X".format(i, pc, 2))
				}
			62 -> return e.sv_q(s)
			63 ->
				when (((i ushr 0) and 67108863)) {
					67043328 -> return e.vnop(s)
					67044128 -> return e.vsync(s)
					67044365 -> return e.vflush(s)
					else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (37) failed mask 0x%08X".format(i, pc, 67108863))
				}
			else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (38) failed mask 0x%08X".format(i, pc, -67108864))
		}
	}
}

fun InstructionDispatcher<CpuState>.dispatch(s: CpuState) {
	s.IR = s.mem.lw(s._PC)
	this.dispatch(s, s._PC, s.IR)
}