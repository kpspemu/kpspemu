package com.soywiz.kpspemu.cpu

import com.soywiz.korio.lang.format

@Suppress("RemoveRedundantCallsOfConversionMethods", "LiftReturnOrAssignment", "RedundantUnitReturnType")
class InstructionDispatcher<T>(val e: InstructionEvaluator<T>) {
	fun dispatch(s: T, pc: Int, i: Int): Unit {
		when ((i and -67108864)) {
			0 ->
				when ((i and 63)) {
					32 -> return e.add(s)
					33 -> return e.addu(s)
					34 -> return e.sub(s)
					35 -> return e.subu(s)
					36 -> return e.and(s)
					39 -> return e.nor(s)
					37 -> return e.or(s)
					38 -> return e.xor(s)
					0 -> return e.sll(s)
					4 -> return e.sllv(s)
					3 -> return e.sra(s)
					7 -> return e.srav(s)
					2 ->
						when ((i and -2097152)) {
							0 -> return e.srl(s)
							2097152 -> return e.rotr(s)
							else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (0) failed mask 0x%08X".format(i, pc, -2097152))
						}
					6 ->
						when ((i and -67106880)) {
							0 -> return e.srlv(s)
							64 -> return e.rotrv(s)
							else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (1) failed mask 0x%08X".format(i, pc, -67106880))
						}
					42 -> return e.slt(s)
					43 -> return e.sltu(s)
					44 -> return e.max(s)
					45 -> return e.min(s)
					26 -> return e.div(s)
					27 -> return e.divu(s)
					24 -> return e.mult(s)
					25 -> return e.multu(s)
					28 -> return e.madd(s)
					29 -> return e.maddu(s)
					46 -> return e.msub(s)
					47 -> return e.msubu(s)
					16 -> return e.mfhi(s)
					18 -> return e.mflo(s)
					17 -> return e.mthi(s)
					19 -> return e.mtlo(s)
					10 -> return e.movz(s)
					11 -> return e.movn(s)
					22 -> return e.clz(s)
					23 -> return e.clo(s)
					8 -> return e.jr(s)
					9 -> return e.jalr(s)
					12 -> return e.syscall(s)
					15 -> return e.sync(s)
					13 -> return e._break(s)
					else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (2) failed mask 0x%08X".format(i, pc, 63))
				}
			536870912 -> return e.addi(s)
			603979776 -> return e.addiu(s)
			805306368 -> return e.andi(s)
			872415232 -> return e.ori(s)
			939524096 -> return e.xori(s)
			671088640 -> return e.slti(s)
			738197504 -> return e.sltiu(s)
			1006632960 -> return e.lui(s)
			2080374784 ->
				when ((i and 63)) {
					32 ->
						when ((i and -2095168)) {
							2080375808 -> return e.seb(s)
							2080376320 -> return e.seh(s)
							2080376064 -> return e.bitrev(s)
							2080374912 -> return e.wsbh(s)
							2080374976 -> return e.wsbw(s)
							else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (3) failed mask 0x%08X".format(i, pc, -2095168))
						}
					0 -> return e.ext(s)
					4 -> return e.ins(s)
					else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (4) failed mask 0x%08X".format(i, pc, 63))
				}
			268435456 -> return e.beq(s)
			1342177280 -> return e.beql(s)
			67108864 ->
				when ((i and 2031616)) {
					65536 -> return e.bgez(s)
					196608 -> return e.bgezl(s)
					1114112 -> return e.bgezal(s)
					1245184 -> return e.bgezall(s)
					0 -> return e.bltz(s)
					131072 -> return e.bltzl(s)
					1048576 -> return e.bltzal(s)
					1179648 -> return e.bltzall(s)
					else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (5) failed mask 0x%08X".format(i, pc, 2031616))
				}
			402653184 -> return e.blez(s)
			1476395008 -> return e.blezl(s)
			469762048 -> return e.bgtz(s)
			1543503872 -> return e.bgtzl(s)
			335544320 -> return e.bne(s)
			1409286144 -> return e.bnel(s)
			134217728 -> return e.j(s)
			201326592 -> return e.jal(s)
			1140850688 ->
				when ((i and 65011712)) {
					16777216 ->
						when ((i and -65077248)) {
							1140850688 -> return e.bc1f(s)
							1140916224 -> return e.bc1t(s)
							1140981760 -> return e.bc1fl(s)
							1141047296 -> return e.bc1tl(s)
							else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (6) failed mask 0x%08X".format(i, pc, -65077248))
						}
					33554432 ->
						when ((i and -67108801)) {
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
							else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (7) failed mask 0x%08X".format(i, pc, -67108801))
						}
					41943040 -> return e.cvt_s_w(s)
					0 -> return e.mfc1(s)
					8388608 -> return e.mtc1(s)
					4194304 -> return e.cfc1(s)
					12582912 -> return e.ctc1(s)
					else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (8) failed mask 0x%08X".format(i, pc, 65011712))
				}
			-2147483648 -> return e.lb(s)
			-2080374784 -> return e.lh(s)
			-1946157056 -> return e.lw(s)
			-2013265920 -> return e.lwl(s)
			-1744830464 -> return e.lwr(s)
			-1879048192 -> return e.lbu(s)
			-1811939328 -> return e.lhu(s)
			-1610612736 -> return e.sb(s)
			-1543503872 -> return e.sh(s)
			-1409286144 -> return e.sw(s)
			-1476395008 -> return e.swl(s)
			-1207959552 -> return e.swr(s)
			-1073741824 -> return e.ll(s)
			-536870912 -> return e.sc(s)
			-1006632960 -> return e.lwc1(s)
			-469762048 -> return e.swc1(s)
			-1140850688 -> return e.cache(s)
			1879048192 ->
				when ((i and 2047)) {
					63 -> return e.dbreak(s)
					0 -> return e.halt(s)
					62 -> return e.dret(s)
					36 -> return e.mfic(s)
					38 -> return e.mtic(s)
					61 ->
						when ((i and -2097152)) {
							1879048192 -> return e.mfdr(s)
							1887436800 -> return e.mtdr(s)
							else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (9) failed mask 0x%08X".format(i, pc, -2097152))
						}
					else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (10) failed mask 0x%08X".format(i, pc, 2047))
				}
			1073741824 ->
				when ((i and 65013759)) {
					33554456 -> return e.eret(s)
					4194304 -> return e.cfc0(s)
					12582912 -> return e.ctc0(s)
					0 -> return e.mfc0(s)
					8388608 -> return e.mtc0(s)
					else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (11) failed mask 0x%08X".format(i, pc, 65013759))
				}
			1207959552 ->
				when ((i and 65011712)) {
					6291456 ->
						when ((i and -67043456)) {
							1207959552 -> return e.mfv(s)
							1207959680 -> return e.mfvc(s)
							else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (12) failed mask 0x%08X".format(i, pc, -67043456))
						}
					14680064 ->
						when ((i and -67043456)) {
							1207959552 -> return e.mtv(s)
							1207959680 -> return e.mtvc(s)
							else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (13) failed mask 0x%08X".format(i, pc, -67043456))
						}
					16777216 ->
						when ((i and -66912256)) {
							1207959552 -> return e.bvf(s)
							1208025088 -> return e.bvt(s)
							1208090624 -> return e.bvfl(s)
							1208156160 -> return e.bvtl(s)
							else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (14) failed mask 0x%08X".format(i, pc, -66912256))
						}
					else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (15) failed mask 0x%08X".format(i, pc, 65011712))
				}
			-939524096 -> return e.lv_s(s)
			-671088640 -> return e.lv_q(s)
			-738197504 ->
				when ((i and 2)) {
					0 -> return e.lvl_q(s)
					2 -> return e.lvr_q(s)
					else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (16) failed mask 0x%08X".format(i, pc, 2))
				}
			-134217728 -> return e.sv_q(s)
			1677721600 ->
				when ((i and 58720256)) {
					8388608 -> return e.vdot(s)
					16777216 -> return e.vscl(s)
					33554432 -> return e.vhdp(s)
					41943040 -> return e.vcrs_t(s)
					0 -> return e.vmul(s)
					50331648 -> return e.vdet(s)
					else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (17) failed mask 0x%08X".format(i, pc, 58720256))
				}
			1811939328 ->
				when ((i and 58720256)) {
					50331648 -> return e.vsge(s)
					58720256 -> return e.vslt(s)
					16777216 -> return e.vmin(s)
					25165824 -> return e.vmax(s)
					0 -> return e.vcmp(s)
					41943040 -> return e.vscmp(s)
					else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (18) failed mask 0x%08X".format(i, pc, 58720256))
				}
			-268435456 ->
				when ((i and 58720256)) {
					58720256 ->
						when ((i and -60817408)) {
							-266338304 -> return e.vrot(s)
							-268435456 ->
								when ((i and 60751872)) {
									58916864 -> return e.vmidt(s)
									58720256 -> return e.vmmov(s)
									59113472 -> return e.vmzero(s)
									59179008 -> return e.vmone(s)
									else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (19) failed mask 0x%08X".format(i, pc, 60751872))
								}
							else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (20) failed mask 0x%08X".format(i, pc, -60817408))
						}
					0 -> return e.vmmul(s)
					41943040 ->
						when ((i and -67075968)) {
							-268402688 -> return e.vcrsp_t(s)
							-268402560 -> return e.vqmul(s)
							else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (21) failed mask 0x%08X".format(i, pc, -67075968))
						}
					8388608 ->
						when ((i and -67075968)) {
							-268435328 -> return e.vtfm2(s)
							-268435456 -> return e.vhtfm2(s)
							else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (22) failed mask 0x%08X".format(i, pc, -67075968))
						}
					16777216 ->
						when ((i and -67075968)) {
							-268402688 -> return e.vtfm3(s)
							-268435328 -> return e.vhtfm3(s)
							else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (23) failed mask 0x%08X".format(i, pc, -67075968))
						}
					25165824 ->
						when ((i and -67075968)) {
							-268402560 -> return e.vtfm4(s)
							-268402688 -> return e.vhtfm4(s)
							else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (24) failed mask 0x%08X".format(i, pc, -67075968))
						}
					33554432 -> return e.vmscl(s)
					else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (25) failed mask 0x%08X".format(i, pc, 58720256))
				}
			-805306368 ->
				when ((i and 50331648)) {
					0 ->
						when ((i and -52428800)) {
							-805306368 ->
								when ((i and 52363264)) {
									393216 -> return e.vzero(s)
									458752 -> return e.vone(s)
									0 -> return e.vmov(s)
									65536 -> return e.vabs(s)
									131072 -> return e.vneg(s)
									1048576 -> return e.vrcp(s)
									1114112 -> return e.vrsq(s)
									1179648 -> return e.vsin(s)
									1245184 -> return e.vcos(s)
									1310720 -> return e.vexp2(s)
									1376256 -> return e.vlog2(s)
									1441792 -> return e.vsqrt(s)
									1507328 -> return e.vasin(s)
									1572864 -> return e.vnrcp(s)
									1703936 -> return e.vnsin(s)
									1835008 -> return e.vrexp2(s)
									262144 -> return e.vsat0(s)
									327680 -> return e.vsat1(s)
									196608 -> return e.vidt(s)
									else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (26) failed mask 0x%08X".format(i, pc, 52363264))
								}
							-801112064 ->
								when ((i and 52363264)) {
									262144 -> return e.vocp(s)
									655360 -> return e.vsgn(s)
									524288 -> return e.vsrt3(s)
									393216 -> return e.vfad(s)
									458752 -> return e.vavg(s)
									1638400 -> return e.vt4444_q(s)
									1703936 -> return e.vt5551_q(s)
									1769472 -> return e.vt5650_q(s)
									1048576 -> return e.vmfvc(s)
									1114112 -> return e.vmtvc(s)
									131072 -> return e.vbfy1(s)
									196608 -> return e.vbfy2(s)
									327680 -> return e.vsocp(s)
									0 -> return e.vsrt1(s)
									65536 -> return e.vsrt2(s)
									589824 -> return e.vsrt4(s)
									else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (27) failed mask 0x%08X".format(i, pc, 52363264))
								}
							-799014912 -> return e.vcst(s)
							-803209216 ->
								when ((i and 52363264)) {
									1900544 -> return e.vi2c(s)
									1835008 -> return e.vi2uc(s)
									0 -> return e.vrnds(s)
									65536 -> return e.vrndi(s)
									131072 -> return e.vrndf1(s)
									196608 -> return e.vrndf2(s)
									1179648 -> return e.vf2h(s)
									1245184 -> return e.vh2f(s)
									2031616 -> return e.vi2s(s)
									1966080 -> return e.vi2us(s)
									1507328 -> return e.vlgb(s)
									1769472 -> return e.vs2i(s)
									1638400 -> return e.vc2i(s)
									1572864 -> return e.vuc2i(s)
									1441792 -> return e.vsbz(s)
									1703936 -> return e.vus2i(s)
									else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (28) failed mask 0x%08X".format(i, pc, 52363264))
								}
							else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (29) failed mask 0x%08X".format(i, pc, -52428800))
						}
					33554432 ->
						when ((i and -52428800)) {
							-794820608 ->
								when ((i and 51904512)) {
									34078720 -> return e.vcmovf(s)
									33554432 -> return e.vcmovt(s)
									else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (30) failed mask 0x%08X".format(i, pc, 51904512))
								}
							-799014912 -> return e.vf2id(s)
							-805306368 -> return e.vf2in(s)
							-801112064 -> return e.vf2iu(s)
							-803209216 -> return e.vf2iz(s)
							-796917760 -> return e.vi2f(s)
							else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (31) failed mask 0x%08X".format(i, pc, -52428800))
						}
					50331648 -> return e.vwbn(s)
					else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (32) failed mask 0x%08X".format(i, pc, 50331648))
				}
			1610612736 ->
				when ((i and 58720256)) {
					0 -> return e.vadd(s)
					8388608 -> return e.vsub(s)
					58720256 -> return e.vdiv(s)
					16777216 -> return e.vsbn(s)
					else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (33) failed mask 0x%08X".format(i, pc, 58720256))
				}
			-603979776 ->
				when ((i and 50331648)) {
					50331648 ->
						when ((i and -58720256)) {
							-603979776 -> return e.viim(s)
							-595591168 -> return e.vfim(s)
							else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (34) failed mask 0x%08X".format(i, pc, -58720256))
						}
					33554432 -> return e.vpfxd(s)
					0 -> return e.vpfxs(s)
					16777216 -> return e.vpfxt(s)
					else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (35) failed mask 0x%08X".format(i, pc, 50331648))
				}
			-67108864 ->
				when ((i and 67108863)) {
					67043328 -> return e.vnop(s)
					67044128 -> return e.vsync(s)
					67044365 -> return e.vflush(s)
					else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (36) failed mask 0x%08X".format(i, pc, 67108863))
				}
			1744830464 -> return e.mfvme(s)
			-1342177280 -> return e.mtvme(s)
			-402653184 -> return e.sv_s(s)
			-201326592 ->
				when ((i and 2)) {
					0 -> return e.svl_q(s)
					2 -> return e.svr_q(s)
					else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (37) failed mask 0x%08X".format(i, pc, 2))
				}
			else -> throw Exception("Invalid instruction 0x%08X at 0x%08X (38) failed mask 0x%08X".format(i, pc, -67108864))
		}
	}
}

fun InstructionDispatcher<CpuState>.dispatch(s: CpuState) {
	s.IR = s.mem.lw(s._PC)
	this.dispatch(s, s._PC, s.IR)
}