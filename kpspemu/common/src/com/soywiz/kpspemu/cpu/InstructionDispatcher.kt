package com.soywiz.kpspemu.cpu

import com.soywiz.korio.lang.*
import com.soywiz.kpspemu.mem.*

@Suppress("RemoveRedundantCallsOfConversionMethods", "LiftReturnOrAssignment", "RedundantUnitReturnType")
class InstructionDispatcher<T>(val e: InstructionEvaluator<T>) {
    fun dispatch(s: T, pc: Int, i: Int): Unit {
        when (((i shr 26) and 63)) {
            0 ->
                when (((i shr 0) and 63)) {
                    0 -> e.sll(s)
                    2 ->
                        when (((i shr 21) and 2047)) {
                            0 -> e.srl(s)
                            1 -> e.rotr(s)
                            else -> throw Exception(
                                "Invalid instruction 0x%08X at 0x%08X (0) failed mask 0x%08X".format(
                                    i,
                                    pc,
                                    -2097152
                                )
                            )
                        }
                    3 -> e.sra(s)
                    4 -> e.sllv(s)
                    6 ->
                        when (((i shr 6) and 66060319)) {
                            0 -> e.srlv(s)
                            1 -> e.rotrv(s)
                            else -> throw Exception(
                                "Invalid instruction 0x%08X at 0x%08X (1) failed mask 0x%08X".format(
                                    i,
                                    pc,
                                    -67106880
                                )
                            )
                        }
                    7 -> e.srav(s)
                    8 -> e.jr(s)
                    9 -> e.jalr(s)
                    10 -> e.movz(s)
                    11 -> e.movn(s)
                    12 -> e.syscall(s)
                    13 -> e._break(s)
                    15 -> e.sync(s)
                    16 -> e.mfhi(s)
                    17 -> e.mthi(s)
                    18 -> e.mflo(s)
                    19 -> e.mtlo(s)
                    22 -> e.clz(s)
                    23 -> e.clo(s)
                    24 -> e.mult(s)
                    25 -> e.multu(s)
                    26 -> e.div(s)
                    27 -> e.divu(s)
                    28 -> e.madd(s)
                    29 -> e.maddu(s)
                    32 -> e.add(s)
                    33 -> e.addu(s)
                    34 -> e.sub(s)
                    35 -> e.subu(s)
                    36 -> e.and(s)
                    37 -> e.or(s)
                    38 -> e.xor(s)
                    39 -> e.nor(s)
                    42 -> e.slt(s)
                    43 -> e.sltu(s)
                    44 -> e.max(s)
                    45 -> e.min(s)
                    46 -> e.msub(s)
                    47 -> e.msubu(s)
                    else -> throw Exception(
                        "Invalid instruction 0x%08X at 0x%08X (2) failed mask 0x%08X".format(
                            i,
                            pc,
                            63
                        )
                    )
                }
            1 ->
                when (((i shr 16) and 31)) {
                    0 -> e.bltz(s)
                    1 -> e.bgez(s)
                    2 -> e.bltzl(s)
                    3 -> e.bgezl(s)
                    16 -> e.bltzal(s)
                    17 -> e.bgezal(s)
                    18 -> e.bltzall(s)
                    19 -> e.bgezall(s)
                    else -> throw Exception(
                        "Invalid instruction 0x%08X at 0x%08X (3) failed mask 0x%08X".format(
                            i,
                            pc,
                            2031616
                        )
                    )
                }
            2 -> e.j(s)
            3 -> e.jal(s)
            4 -> e.beq(s)
            5 -> e.bne(s)
            6 -> e.blez(s)
            7 -> e.bgtz(s)
            8 -> e.addi(s)
            9 -> e.addiu(s)
            10 -> e.slti(s)
            11 -> e.sltiu(s)
            12 -> e.andi(s)
            13 -> e.ori(s)
            14 -> e.xori(s)
            15 -> e.lui(s)
            16 ->
                when (((i shr 0) and 65013759)) {
                    0 -> e.mfc0(s)
                    4194304 -> e.cfc0(s)
                    8388608 -> e.mtc0(s)
                    12582912 -> e.ctc0(s)
                    33554456 -> e.eret(s)
                    else -> throw Exception(
                        "Invalid instruction 0x%08X at 0x%08X (4) failed mask 0x%08X".format(
                            i,
                            pc,
                            65013759
                        )
                    )
                }
            17 ->
                when (((i shr 21) and 31)) {
                    0 -> e.mfc1(s)
                    2 -> e.cfc1(s)
                    4 -> e.mtc1(s)
                    6 -> e.ctc1(s)
                    8 ->
                        when (((i shr 16) and 64543)) {
                            17408 -> e.bc1f(s)
                            17409 -> e.bc1t(s)
                            17410 -> e.bc1fl(s)
                            17411 -> e.bc1tl(s)
                            else -> throw Exception(
                                "Invalid instruction 0x%08X at 0x%08X (5) failed mask 0x%08X".format(
                                    i,
                                    pc,
                                    -65077248
                                )
                            )
                        }
                    16 ->
                        when (((i shr 0) and -67108801)) {
                            1140850688 -> e.add_s(s)
                            1140850689 -> e.sub_s(s)
                            1140850690 -> e.mul_s(s)
                            1140850691 -> e.div_s(s)
                            1140850692 -> e.sqrt_s(s)
                            1140850693 -> e.abs_s(s)
                            1140850694 -> e.mov_s(s)
                            1140850695 -> e.neg_s(s)
                            1140850700 -> e.round_w_s(s)
                            1140850701 -> e.trunc_w_s(s)
                            1140850702 -> e.ceil_w_s(s)
                            1140850703 -> e.floor_w_s(s)
                            1140850724 -> e.cvt_w_s(s)
                            1140850736 -> e.c_f_s(s)
                            1140850737 -> e.c_un_s(s)
                            1140850738 -> e.c_eq_s(s)
                            1140850739 -> e.c_ueq_s(s)
                            1140850740 -> e.c_olt_s(s)
                            1140850741 -> e.c_ult_s(s)
                            1140850742 -> e.c_ole_s(s)
                            1140850743 -> e.c_ule_s(s)
                            1140850744 -> e.c_sf_s(s)
                            1140850745 -> e.c_ngle_s(s)
                            1140850746 -> e.c_seq_s(s)
                            1140850747 -> e.c_ngl_s(s)
                            1140850748 -> e.c_lt_s(s)
                            1140850749 -> e.c_nge_s(s)
                            1140850750 -> e.c_le_s(s)
                            1140850751 -> e.c_ngt_s(s)
                            else -> throw Exception(
                                "Invalid instruction 0x%08X at 0x%08X (6) failed mask 0x%08X".format(
                                    i,
                                    pc,
                                    -67108801
                                )
                            )
                        }
                    20 -> e.cvt_s_w(s)
                    else -> throw Exception(
                        "Invalid instruction 0x%08X at 0x%08X (7) failed mask 0x%08X".format(
                            i,
                            pc,
                            65011712
                        )
                    )
                }
            18 ->
                when (((i shr 21) and 31)) {
                    3 ->
                        when (((i shr 7) and 33030655)) {
                            9437184 -> e.mfv(s)
                            9437185 -> e.mfvc(s)
                            else -> throw Exception(
                                "Invalid instruction 0x%08X at 0x%08X (8) failed mask 0x%08X".format(
                                    i,
                                    pc,
                                    -67043456
                                )
                            )
                        }
                    7 ->
                        when (((i shr 7) and 33030655)) {
                            9437184 -> e.mtv(s)
                            9437185 -> e.mtvc(s)
                            else -> throw Exception(
                                "Invalid instruction 0x%08X at 0x%08X (9) failed mask 0x%08X".format(
                                    i,
                                    pc,
                                    -67043456
                                )
                            )
                        }
                    8 ->
                        when (((i shr 16) and 64515)) {
                            18432 -> e.bvf(s)
                            18433 -> e.bvt(s)
                            18434 -> e.bvfl(s)
                            18435 -> e.bvtl(s)
                            else -> throw Exception(
                                "Invalid instruction 0x%08X at 0x%08X (10) failed mask 0x%08X".format(
                                    i,
                                    pc,
                                    -66912256
                                )
                            )
                        }
                    else -> throw Exception(
                        "Invalid instruction 0x%08X at 0x%08X (11) failed mask 0x%08X".format(
                            i,
                            pc,
                            65011712
                        )
                    )
                }
            20 -> e.beql(s)
            21 -> e.bnel(s)
            22 -> e.blezl(s)
            23 -> e.bgtzl(s)
            24 ->
                when (((i shr 23) and 7)) {
                    0 -> e.vadd(s)
                    1 -> e.vsub(s)
                    2 -> e.vsbn(s)
                    7 -> e.vdiv(s)
                    else -> throw Exception(
                        "Invalid instruction 0x%08X at 0x%08X (12) failed mask 0x%08X".format(
                            i,
                            pc,
                            58720256
                        )
                    )
                }
            25 ->
                when (((i shr 23) and 7)) {
                    0 -> e.vmul(s)
                    1 -> e.vdot(s)
                    2 -> e.vscl(s)
                    4 -> e.vhdp(s)
                    5 -> e.vcrs_t(s)
                    6 -> e.vdet(s)
                    else -> throw Exception(
                        "Invalid instruction 0x%08X at 0x%08X (13) failed mask 0x%08X".format(
                            i,
                            pc,
                            58720256
                        )
                    )
                }
            26 -> e.mfvme(s)
            27 ->
                when (((i shr 23) and 7)) {
                    0 -> e.vcmp(s)
                    2 -> e.vmin(s)
                    3 -> e.vmax(s)
                    5 -> e.vscmp(s)
                    6 -> e.vsge(s)
                    7 -> e.vslt(s)
                    else -> throw Exception(
                        "Invalid instruction 0x%08X at 0x%08X (14) failed mask 0x%08X".format(
                            i,
                            pc,
                            58720256
                        )
                    )
                }
            28 ->
                when (((i shr 0) and 2047)) {
                    0 -> e.halt(s)
                    36 -> e.mfic(s)
                    38 -> e.mtic(s)
                    61 ->
                        when (((i shr 21) and 2047)) {
                            896 -> e.mfdr(s)
                            900 -> e.mtdr(s)
                            else -> throw Exception(
                                "Invalid instruction 0x%08X at 0x%08X (15) failed mask 0x%08X".format(
                                    i,
                                    pc,
                                    -2097152
                                )
                            )
                        }
                    62 -> e.dret(s)
                    63 -> e.dbreak(s)
                    else -> throw Exception(
                        "Invalid instruction 0x%08X at 0x%08X (16) failed mask 0x%08X".format(
                            i,
                            pc,
                            2047
                        )
                    )
                }
            31 ->
                when (((i shr 0) and 63)) {
                    0 -> e.ext(s)
                    4 -> e.ins(s)
                    32 ->
                        when (((i shr 6) and 67076127)) {
                            32505858 -> e.wsbh(s)
                            32505859 -> e.wsbw(s)
                            32505872 -> e.seb(s)
                            32505876 -> e.bitrev(s)
                            32505880 -> e.seh(s)
                            else -> throw Exception(
                                "Invalid instruction 0x%08X at 0x%08X (17) failed mask 0x%08X".format(
                                    i,
                                    pc,
                                    -2095168
                                )
                            )
                        }
                    else -> throw Exception(
                        "Invalid instruction 0x%08X at 0x%08X (18) failed mask 0x%08X".format(
                            i,
                            pc,
                            63
                        )
                    )
                }
            32 -> e.lb(s)
            33 -> e.lh(s)
            34 -> e.lwl(s)
            35 -> e.lw(s)
            36 -> e.lbu(s)
            37 -> e.lhu(s)
            38 -> e.lwr(s)
            40 -> e.sb(s)
            41 -> e.sh(s)
            42 -> e.swl(s)
            43 -> e.sw(s)
            44 -> e.mtvme(s)
            46 -> e.swr(s)
            47 -> e.cache(s)
            48 -> e.ll(s)
            49 -> e.lwc1(s)
            50 -> e.lv_s(s)
            52 ->
                when (((i shr 24) and 3)) {
                    0 ->
                        when (((i shr 21) and 2023)) {
                            1664 ->
                                when (((i shr 16) and 799)) {
                                    0 -> e.vmov(s)
                                    1 -> e.vabs(s)
                                    2 -> e.vneg(s)
                                    3 -> e.vidt(s)
                                    4 -> e.vsat0(s)
                                    5 -> e.vsat1(s)
                                    6 -> e.vzero(s)
                                    7 -> e.vone(s)
                                    16 -> e.vrcp(s)
                                    17 -> e.vrsq(s)
                                    18 -> e.vsin(s)
                                    19 -> e.vcos(s)
                                    20 -> e.vexp2(s)
                                    21 -> e.vlog2(s)
                                    22 -> e.vsqrt(s)
                                    23 -> e.vasin(s)
                                    24 -> e.vnrcp(s)
                                    26 -> e.vnsin(s)
                                    28 -> e.vrexp2(s)
                                    else -> throw Exception(
                                        "Invalid instruction 0x%08X at 0x%08X (19) failed mask 0x%08X".format(
                                            i,
                                            pc,
                                            52363264
                                        )
                                    )
                                }
                            1665 ->
                                when (((i shr 16) and 799)) {
                                    0 -> e.vrnds(s)
                                    1 -> e.vrndi(s)
                                    2 -> e.vrndf1(s)
                                    3 -> e.vrndf2(s)
                                    18 -> e.vf2h(s)
                                    19 -> e.vh2f(s)
                                    22 -> e.vsbz(s)
                                    23 -> e.vlgb(s)
                                    24 -> e.vuc2i(s)
                                    25 -> e.vc2i(s)
                                    26 -> e.vus2i(s)
                                    27 -> e.vs2i(s)
                                    28 -> e.vi2uc(s)
                                    29 -> e.vi2c(s)
                                    30 -> e.vi2us(s)
                                    31 -> e.vi2s(s)
                                    else -> throw Exception(
                                        "Invalid instruction 0x%08X at 0x%08X (20) failed mask 0x%08X".format(
                                            i,
                                            pc,
                                            52363264
                                        )
                                    )
                                }
                            1666 ->
                                when (((i shr 16) and 799)) {
                                    0 -> e.vsrt1(s)
                                    1 -> e.vsrt2(s)
                                    2 -> e.vbfy1(s)
                                    3 -> e.vbfy2(s)
                                    4 -> e.vocp(s)
                                    5 -> e.vsocp(s)
                                    6 -> e.vfad(s)
                                    7 -> e.vavg(s)
                                    8 -> e.vsrt3(s)
                                    9 -> e.vsrt4(s)
                                    10 -> e.vsgn(s)
                                    16 -> e.vmfvc(s)
                                    17 -> e.vmtvc(s)
                                    25 -> e.vt4444_q(s)
                                    26 -> e.vt5551_q(s)
                                    27 -> e.vt5650_q(s)
                                    else -> throw Exception(
                                        "Invalid instruction 0x%08X at 0x%08X (21) failed mask 0x%08X".format(
                                            i,
                                            pc,
                                            52363264
                                        )
                                    )
                                }
                            1667 -> e.vcst(s)
                            else -> throw Exception(
                                "Invalid instruction 0x%08X at 0x%08X (22) failed mask 0x%08X".format(
                                    i,
                                    pc,
                                    -52428800
                                )
                            )
                        }
                    2 ->
                        when (((i shr 21) and 2023)) {
                            1664 -> e.vf2in(s)
                            1665 -> e.vf2iz(s)
                            1666 -> e.vf2iu(s)
                            1667 -> e.vf2id(s)
                            1668 -> e.vi2f(s)
                            1669 ->
                                when (((i shr 19) and 99)) {
                                    64 -> e.vcmovt(s)
                                    65 -> e.vcmovf(s)
                                    else -> throw Exception(
                                        "Invalid instruction 0x%08X at 0x%08X (23) failed mask 0x%08X".format(
                                            i,
                                            pc,
                                            51904512
                                        )
                                    )
                                }
                            else -> throw Exception(
                                "Invalid instruction 0x%08X at 0x%08X (24) failed mask 0x%08X".format(
                                    i,
                                    pc,
                                    -52428800
                                )
                            )
                        }
                    3 -> e.vwbn(s)
                    else -> throw Exception(
                        "Invalid instruction 0x%08X at 0x%08X (25) failed mask 0x%08X".format(
                            i,
                            pc,
                            50331648
                        )
                    )
                }
            53 ->
                when (((i shr 1) and 1)) {
                    0 -> e.lvl_q(s)
                    1 -> e.lvr_q(s)
                    else -> throw Exception(
                        "Invalid instruction 0x%08X at 0x%08X (26) failed mask 0x%08X".format(
                            i,
                            pc,
                            2
                        )
                    )
                }
            54 -> e.lv_q(s)
            55 ->
                when (((i shr 24) and 3)) {
                    0 -> e.vpfxs(s)
                    1 -> e.vpfxt(s)
                    2 -> e.vpfxd(s)
                    3 ->
                        when (((i shr 23) and 505)) {
                            440 -> e.viim(s)
                            441 -> e.vfim(s)
                            else -> throw Exception(
                                "Invalid instruction 0x%08X at 0x%08X (27) failed mask 0x%08X".format(
                                    i,
                                    pc,
                                    -58720256
                                )
                            )
                        }
                    else -> throw Exception(
                        "Invalid instruction 0x%08X at 0x%08X (28) failed mask 0x%08X".format(
                            i,
                            pc,
                            50331648
                        )
                    )
                }
            56 -> e.sc(s)
            57 -> e.swc1(s)
            58 -> e.sv_s(s)
            60 ->
                when (((i shr 23) and 7)) {
                    0 -> e.vmmul(s)
                    1 ->
                        when (((i shr 7) and 33030401)) {
                            31457280 -> e.vhtfm2(s)
                            31457281 -> e.vtfm2(s)
                            else -> throw Exception(
                                "Invalid instruction 0x%08X at 0x%08X (29) failed mask 0x%08X".format(
                                    i,
                                    pc,
                                    -67075968
                                )
                            )
                        }
                    2 ->
                        when (((i shr 7) and 33030401)) {
                            31457281 -> e.vhtfm3(s)
                            31457536 -> e.vtfm3(s)
                            else -> throw Exception(
                                "Invalid instruction 0x%08X at 0x%08X (30) failed mask 0x%08X".format(
                                    i,
                                    pc,
                                    -67075968
                                )
                            )
                        }
                    3 ->
                        when (((i shr 7) and 33030401)) {
                            31457536 -> e.vhtfm4(s)
                            31457537 -> e.vtfm4(s)
                            else -> throw Exception(
                                "Invalid instruction 0x%08X at 0x%08X (31) failed mask 0x%08X".format(
                                    i,
                                    pc,
                                    -67075968
                                )
                            )
                        }
                    4 -> e.vmscl(s)
                    5 ->
                        when (((i shr 7) and 33030401)) {
                            31457536 -> e.vcrsp_t(s)
                            31457537 -> e.vqmul(s)
                            else -> throw Exception(
                                "Invalid instruction 0x%08X at 0x%08X (32) failed mask 0x%08X".format(
                                    i,
                                    pc,
                                    -67075968
                                )
                            )
                        }
                    7 ->
                        when (((i shr 21) and 2019)) {
                            1920 ->
                                when (((i shr 16) and 927)) {
                                    896 -> e.vmmov(s)
                                    899 -> e.vmidt(s)
                                    902 -> e.vmzero(s)
                                    903 -> e.vmone(s)
                                    else -> throw Exception(
                                        "Invalid instruction 0x%08X at 0x%08X (33) failed mask 0x%08X".format(
                                            i,
                                            pc,
                                            60751872
                                        )
                                    )
                                }
                            1921 -> e.vrot(s)
                            else -> throw Exception(
                                "Invalid instruction 0x%08X at 0x%08X (34) failed mask 0x%08X".format(
                                    i,
                                    pc,
                                    -60817408
                                )
                            )
                        }
                    else -> throw Exception(
                        "Invalid instruction 0x%08X at 0x%08X (35) failed mask 0x%08X".format(
                            i,
                            pc,
                            58720256
                        )
                    )
                }
            61 ->
                when (((i shr 1) and 1)) {
                    0 -> e.svl_q(s)
                    1 -> e.svr_q(s)
                    else -> throw Exception(
                        "Invalid instruction 0x%08X at 0x%08X (36) failed mask 0x%08X".format(
                            i,
                            pc,
                            2
                        )
                    )
                }
            62 -> e.sv_q(s)
            63 ->
                when (((i shr 0) and 67108863)) {
                    67043328 -> e.vnop(s)
                    67044128 -> e.vsync(s)
                    67044365 -> e.vflush(s)
                    else -> throw Exception(
                        "Invalid instruction 0x%08X at 0x%08X (37) failed mask 0x%08X".format(
                            i,
                            pc,
                            67108863
                        )
                    )
                }
            else -> throw Exception(
                "Invalid instruction 0x%08X at 0x%08X (38) failed mask 0x%08X".format(
                    i,
                    pc,
                    -67108864
                )
            )
        }
    }
}

fun InstructionDispatcher<CpuState>.dispatch(s: CpuState) {
    s.IR = s.mem.lw(s._PC)
    this.dispatch(s, s._PC, s.IR)
}