package com.soywiz.kpspemu.cpu

import com.soywiz.korio.lang.*
import com.soywiz.kpspemu.mem.*

@Suppress("RemoveRedundantCallsOfConversionMethods", "LiftReturnOrAssignment", "RedundantUnitReturnType")
class InstructionDispatcher<T>(val e: InstructionEvaluator<T>) {
    fun dispatch(s: T, pc: Int, i: Int): Unit {
        when (((i shr 26) and 63)) {
            0 ->
                when (((i shr 0) and 63)) {
                    0 -> return e.sll(i, s)
                    2 ->
                        when (((i shr 21) and 2047)) {
                            0 -> return e.srl(i, s)
                            1 -> return e.rotr(i, s)
                            else -> throw Exception(
                                "Invalid instruction 0x%08X at 0x%08X (0) failed mask 0x%08X".format(
                                    i,
                                    pc,
                                    -2097152
                                )
                            )
                        }
                    3 -> return e.sra(i, s)
                    4 -> return e.sllv(i, s)
                    6 ->
                        when (((i shr 6) and 66060319)) {
                            0 -> return e.srlv(i, s)
                            1 -> return e.rotrv(i, s)
                            else -> throw Exception(
                                "Invalid instruction 0x%08X at 0x%08X (1) failed mask 0x%08X".format(
                                    i,
                                    pc,
                                    -67106880
                                )
                            )
                        }
                    7 -> return e.srav(i, s)
                    8 -> return e.jr(i, s)
                    9 -> return e.jalr(i, s)
                    10 -> return e.movz(i, s)
                    11 -> return e.movn(i, s)
                    12 -> return e.syscall(i, s)
                    13 -> return e._break(i, s)
                    15 -> return e.sync(i, s)
                    16 -> return e.mfhi(i, s)
                    17 -> return e.mthi(i, s)
                    18 -> return e.mflo(i, s)
                    19 -> return e.mtlo(i, s)
                    22 -> return e.clz(i, s)
                    23 -> return e.clo(i, s)
                    24 -> return e.mult(i, s)
                    25 -> return e.multu(i, s)
                    26 -> return e.div(i, s)
                    27 -> return e.divu(i, s)
                    28 -> return e.madd(i, s)
                    29 -> return e.maddu(i, s)
                    32 -> return e.add(i, s)
                    33 -> return e.addu(i, s)
                    34 -> return e.sub(i, s)
                    35 -> return e.subu(i, s)
                    36 -> return e.and(i, s)
                    37 -> return e.or(i, s)
                    38 -> return e.xor(i, s)
                    39 -> return e.nor(i, s)
                    42 -> return e.slt(i, s)
                    43 -> return e.sltu(i, s)
                    44 -> return e.max(i, s)
                    45 -> return e.min(i, s)
                    46 -> return e.msub(i, s)
                    47 -> return e.msubu(i, s)
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
                    0 -> return e.bltz(i, s)
                    1 -> return e.bgez(i, s)
                    2 -> return e.bltzl(i, s)
                    3 -> return e.bgezl(i, s)
                    16 -> return e.bltzal(i, s)
                    17 -> return e.bgezal(i, s)
                    18 -> return e.bltzall(i, s)
                    19 -> return e.bgezall(i, s)
                    else -> throw Exception(
                        "Invalid instruction 0x%08X at 0x%08X (3) failed mask 0x%08X".format(
                            i,
                            pc,
                            2031616
                        )
                    )
                }
            2 -> return e.j(i, s)
            3 -> return e.jal(i, s)
            4 -> return e.beq(i, s)
            5 -> return e.bne(i, s)
            6 -> return e.blez(i, s)
            7 -> return e.bgtz(i, s)
            8 -> return e.addi(i, s)
            9 -> return e.addiu(i, s)
            10 -> return e.slti(i, s)
            11 -> return e.sltiu(i, s)
            12 -> return e.andi(i, s)
            13 -> return e.ori(i, s)
            14 -> return e.xori(i, s)
            15 -> return e.lui(i, s)
            16 ->
                when (((i shr 0) and 65013759)) {
                    0 -> return e.mfc0(i, s)
                    4194304 -> return e.cfc0(i, s)
                    8388608 -> return e.mtc0(i, s)
                    12582912 -> return e.ctc0(i, s)
                    33554456 -> return e.eret(i, s)
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
                    0 -> return e.mfc1(i, s)
                    2 -> return e.cfc1(i, s)
                    4 -> return e.mtc1(i, s)
                    6 -> return e.ctc1(i, s)
                    8 ->
                        when (((i shr 16) and 64543)) {
                            17408 -> return e.bc1f(i, s)
                            17409 -> return e.bc1t(i, s)
                            17410 -> return e.bc1fl(i, s)
                            17411 -> return e.bc1tl(i, s)
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
                            1140850688 -> return e.add_s(i, s)
                            1140850689 -> return e.sub_s(i, s)
                            1140850690 -> return e.mul_s(i, s)
                            1140850691 -> return e.div_s(i, s)
                            1140850692 -> return e.sqrt_s(i, s)
                            1140850693 -> return e.abs_s(i, s)
                            1140850694 -> return e.mov_s(i, s)
                            1140850695 -> return e.neg_s(i, s)
                            1140850700 -> return e.round_w_s(i, s)
                            1140850701 -> return e.trunc_w_s(i, s)
                            1140850702 -> return e.ceil_w_s(i, s)
                            1140850703 -> return e.floor_w_s(i, s)
                            1140850724 -> return e.cvt_w_s(i, s)
                            1140850736 -> return e.c_f_s(i, s)
                            1140850737 -> return e.c_un_s(i, s)
                            1140850738 -> return e.c_eq_s(i, s)
                            1140850739 -> return e.c_ueq_s(i, s)
                            1140850740 -> return e.c_olt_s(i, s)
                            1140850741 -> return e.c_ult_s(i, s)
                            1140850742 -> return e.c_ole_s(i, s)
                            1140850743 -> return e.c_ule_s(i, s)
                            1140850744 -> return e.c_sf_s(i, s)
                            1140850745 -> return e.c_ngle_s(i, s)
                            1140850746 -> return e.c_seq_s(i, s)
                            1140850747 -> return e.c_ngl_s(i, s)
                            1140850748 -> return e.c_lt_s(i, s)
                            1140850749 -> return e.c_nge_s(i, s)
                            1140850750 -> return e.c_le_s(i, s)
                            1140850751 -> return e.c_ngt_s(i, s)
                            else -> throw Exception(
                                "Invalid instruction 0x%08X at 0x%08X (6) failed mask 0x%08X".format(
                                    i,
                                    pc,
                                    -67108801
                                )
                            )
                        }
                    20 -> return e.cvt_s_w(i, s)
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
                            9437184 -> return e.mfv(i, s)
                            9437185 -> return e.mfvc(i, s)
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
                            9437184 -> return e.mtv(i, s)
                            9437185 -> return e.mtvc(i, s)
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
                            18432 -> return e.bvf(i, s)
                            18433 -> return e.bvt(i, s)
                            18434 -> return e.bvfl(i, s)
                            18435 -> return e.bvtl(i, s)
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
            20 -> return e.beql(i, s)
            21 -> return e.bnel(i, s)
            22 -> return e.blezl(i, s)
            23 -> return e.bgtzl(i, s)
            24 ->
                when (((i shr 23) and 7)) {
                    0 -> return e.vadd(i, s)
                    1 -> return e.vsub(i, s)
                    2 -> return e.vsbn(i, s)
                    7 -> return e.vdiv(i, s)
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
                    0 -> return e.vmul(i, s)
                    1 -> return e.vdot(i, s)
                    2 -> return e.vscl(i, s)
                    4 -> return e.vhdp(i, s)
                    5 -> return e.vcrs_t(i, s)
                    6 -> return e.vdet(i, s)
                    else -> throw Exception(
                        "Invalid instruction 0x%08X at 0x%08X (13) failed mask 0x%08X".format(
                            i,
                            pc,
                            58720256
                        )
                    )
                }
            26 -> return e.mfvme(i, s)
            27 ->
                when (((i shr 23) and 7)) {
                    0 -> return e.vcmp(i, s)
                    2 -> return e.vmin(i, s)
                    3 -> return e.vmax(i, s)
                    5 -> return e.vscmp(i, s)
                    6 -> return e.vsge(i, s)
                    7 -> return e.vslt(i, s)
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
                    0 -> return e.halt(i, s)
                    36 -> return e.mfic(i, s)
                    38 -> return e.mtic(i, s)
                    61 ->
                        when (((i shr 21) and 2047)) {
                            896 -> return e.mfdr(i, s)
                            900 -> return e.mtdr(i, s)
                            else -> throw Exception(
                                "Invalid instruction 0x%08X at 0x%08X (15) failed mask 0x%08X".format(
                                    i,
                                    pc,
                                    -2097152
                                )
                            )
                        }
                    62 -> return e.dret(i, s)
                    63 -> return e.dbreak(i, s)
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
                    0 -> return e.ext(i, s)
                    4 -> return e.ins(i, s)
                    32 ->
                        when (((i shr 6) and 67076127)) {
                            32505858 -> return e.wsbh(i, s)
                            32505859 -> return e.wsbw(i, s)
                            32505872 -> return e.seb(i, s)
                            32505876 -> return e.bitrev(i, s)
                            32505880 -> return e.seh(i, s)
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
            32 -> return e.lb(i, s)
            33 -> return e.lh(i, s)
            34 -> return e.lwl(i, s)
            35 -> return e.lw(i, s)
            36 -> return e.lbu(i, s)
            37 -> return e.lhu(i, s)
            38 -> return e.lwr(i, s)
            40 -> return e.sb(i, s)
            41 -> return e.sh(i, s)
            42 -> return e.swl(i, s)
            43 -> return e.sw(i, s)
            44 -> return e.mtvme(i, s)
            46 -> return e.swr(i, s)
            47 -> return e.cache(i, s)
            48 -> return e.ll(i, s)
            49 -> return e.lwc1(i, s)
            50 -> return e.lv_s(i, s)
            52 ->
                when (((i shr 24) and 3)) {
                    0 ->
                        when (((i shr 21) and 2023)) {
                            1664 ->
                                when (((i shr 16) and 799)) {
                                    0 -> return e.vmov(i, s)
                                    1 -> return e.vabs(i, s)
                                    2 -> return e.vneg(i, s)
                                    3 -> return e.vidt(i, s)
                                    4 -> return e.vsat0(i, s)
                                    5 -> return e.vsat1(i, s)
                                    6 -> return e.vzero(i, s)
                                    7 -> return e.vone(i, s)
                                    16 -> return e.vrcp(i, s)
                                    17 -> return e.vrsq(i, s)
                                    18 -> return e.vsin(i, s)
                                    19 -> return e.vcos(i, s)
                                    20 -> return e.vexp2(i, s)
                                    21 -> return e.vlog2(i, s)
                                    22 -> return e.vsqrt(i, s)
                                    23 -> return e.vasin(i, s)
                                    24 -> return e.vnrcp(i, s)
                                    26 -> return e.vnsin(i, s)
                                    28 -> return e.vrexp2(i, s)
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
                                    0 -> return e.vrnds(i, s)
                                    1 -> return e.vrndi(i, s)
                                    2 -> return e.vrndf1(i, s)
                                    3 -> return e.vrndf2(i, s)
                                    18 -> return e.vf2h(i, s)
                                    19 -> return e.vh2f(i, s)
                                    22 -> return e.vsbz(i, s)
                                    23 -> return e.vlgb(i, s)
                                    24 -> return e.vuc2i(i, s)
                                    25 -> return e.vc2i(i, s)
                                    26 -> return e.vus2i(i, s)
                                    27 -> return e.vs2i(i, s)
                                    28 -> return e.vi2uc(i, s)
                                    29 -> return e.vi2c(i, s)
                                    30 -> return e.vi2us(i, s)
                                    31 -> return e.vi2s(i, s)
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
                                    0 -> return e.vsrt1(i, s)
                                    1 -> return e.vsrt2(i, s)
                                    2 -> return e.vbfy1(i, s)
                                    3 -> return e.vbfy2(i, s)
                                    4 -> return e.vocp(i, s)
                                    5 -> return e.vsocp(i, s)
                                    6 -> return e.vfad(i, s)
                                    7 -> return e.vavg(i, s)
                                    8 -> return e.vsrt3(i, s)
                                    9 -> return e.vsrt4(i, s)
                                    10 -> return e.vsgn(i, s)
                                    16 -> return e.vmfvc(i, s)
                                    17 -> return e.vmtvc(i, s)
                                    25 -> return e.vt4444_q(i, s)
                                    26 -> return e.vt5551_q(i, s)
                                    27 -> return e.vt5650_q(i, s)
                                    else -> throw Exception(
                                        "Invalid instruction 0x%08X at 0x%08X (21) failed mask 0x%08X".format(
                                            i,
                                            pc,
                                            52363264
                                        )
                                    )
                                }
                            1667 -> return e.vcst(i, s)
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
                            1664 -> return e.vf2in(i, s)
                            1665 -> return e.vf2iz(i, s)
                            1666 -> return e.vf2iu(i, s)
                            1667 -> return e.vf2id(i, s)
                            1668 -> return e.vi2f(i, s)
                            1669 ->
                                when (((i shr 19) and 99)) {
                                    64 -> return e.vcmovt(i, s)
                                    65 -> return e.vcmovf(i, s)
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
                    3 -> return e.vwbn(i, s)
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
                    0 -> return e.lvl_q(i, s)
                    1 -> return e.lvr_q(i, s)
                    else -> throw Exception(
                        "Invalid instruction 0x%08X at 0x%08X (26) failed mask 0x%08X".format(
                            i,
                            pc,
                            2
                        )
                    )
                }
            54 -> return e.lv_q(i, s)
            55 ->
                when (((i shr 24) and 3)) {
                    0 -> return e.vpfxs(i, s)
                    1 -> return e.vpfxt(i, s)
                    2 -> return e.vpfxd(i, s)
                    3 ->
                        when (((i shr 23) and 505)) {
                            440 -> return e.viim(i, s)
                            441 -> return e.vfim(i, s)
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
            56 -> return e.sc(i, s)
            57 -> return e.swc1(i, s)
            58 -> return e.sv_s(i, s)
            60 ->
                when (((i shr 23) and 7)) {
                    0 -> return e.vmmul(i, s)
                    1 ->
                        when (((i shr 7) and 33030401)) {
                            31457280 -> return e.vhtfm2(i, s)
                            31457281 -> return e.vtfm2(i, s)
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
                            31457281 -> return e.vhtfm3(i, s)
                            31457536 -> return e.vtfm3(i, s)
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
                            31457536 -> return e.vhtfm4(i, s)
                            31457537 -> return e.vtfm4(i, s)
                            else -> throw Exception(
                                "Invalid instruction 0x%08X at 0x%08X (31) failed mask 0x%08X".format(
                                    i,
                                    pc,
                                    -67075968
                                )
                            )
                        }
                    4 -> return e.vmscl(i, s)
                    5 ->
                        when (((i shr 7) and 33030401)) {
                            31457536 -> return e.vcrsp_t(i, s)
                            31457537 -> return e.vqmul(i, s)
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
                                    896 -> return e.vmmov(i, s)
                                    899 -> return e.vmidt(i, s)
                                    902 -> return e.vmzero(i, s)
                                    903 -> return e.vmone(i, s)
                                    else -> throw Exception(
                                        "Invalid instruction 0x%08X at 0x%08X (33) failed mask 0x%08X".format(
                                            i,
                                            pc,
                                            60751872
                                        )
                                    )
                                }
                            1921 -> return e.vrot(i, s)
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
                    0 -> return e.svl_q(i, s)
                    1 -> return e.svr_q(i, s)
                    else -> throw Exception(
                        "Invalid instruction 0x%08X at 0x%08X (36) failed mask 0x%08X".format(
                            i,
                            pc,
                            2
                        )
                    )
                }
            62 -> return e.sv_q(i, s)
            63 ->
                when (((i shr 0) and 67108863)) {
                    67043328 -> return e.vnop(i, s)
                    67044128 -> return e.vsync(i, s)
                    67044365 -> return e.vflush(i, s)
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