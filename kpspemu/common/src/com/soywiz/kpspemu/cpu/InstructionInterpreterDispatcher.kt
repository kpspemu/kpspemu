package com.soywiz.kpspemu.cpu

import com.soywiz.korio.lang.*
import com.soywiz.kpspemu.cpu.interpreter.*
import com.soywiz.kpspemu.mem.*

@Suppress("RemoveRedundantCallsOfConversionMethods", "LiftReturnOrAssignment", "RedundantUnitReturnType")
class InstructionInterpreterDispatcher {
    inline fun dispatch(e: InstructionInterpreter, s: CpuState, r: CpuRegisters, m: Memory, pc: Int, i: Int): Unit {
        when (((i shr 26) and 63)) {
            0 ->
                when (((i shr 0) and 63)) {
                    0 -> e.sll(s, r, m)
                    2 ->
                        when (((i shr 21) and 2047)) {
                            0 -> e.srl(s, r, m)
                            1 -> e.rotr(s, r, m)
                            else -> throw Exception(
                                "Invalid instruction 0x%08X at 0x%08X (0) failed mask 0x%08X".format(
                                    i,
                                    pc,
                                    -2097152
                                )
                            )
                        }
                    3 -> e.sra(s, r, m)
                    4 -> e.sllv(s, r, m)
                    6 ->
                        when (((i shr 6) and 66060319)) {
                            0 -> e.srlv(s, r, m)
                            1 -> e.rotrv(s, r, m)
                            else -> throw Exception(
                                "Invalid instruction 0x%08X at 0x%08X (1) failed mask 0x%08X".format(
                                    i,
                                    pc,
                                    -67106880
                                )
                            )
                        }
                    7 -> e.srav(s, r, m)
                    8 -> e.jr(s, r, m)
                    9 -> e.jalr(s, r, m)
                    10 -> e.movz(s, r, m)
                    11 -> e.movn(s, r, m)
                    12 -> e.syscall(s, r, m)
                    13 -> e._break(s, r, m)
                    15 -> e.sync(s, r, m)
                    16 -> e.mfhi(s, r, m)
                    17 -> e.mthi(s, r, m)
                    18 -> e.mflo(s, r, m)
                    19 -> e.mtlo(s, r, m)
                    22 -> e.clz(s, r, m)
                    23 -> e.clo(s, r, m)
                    24 -> e.mult(s, r, m)
                    25 -> e.multu(s, r, m)
                    26 -> e.div(s, r, m)
                    27 -> e.divu(s, r, m)
                    28 -> e.madd(s, r, m)
                    29 -> e.maddu(s, r, m)
                    32 -> e.add(s, r, m)
                    33 -> e.addu(s, r, m)
                    34 -> e.sub(s, r, m)
                    35 -> e.subu(s, r, m)
                    36 -> e.and(s, r, m)
                    37 -> e.or(s, r, m)
                    38 -> e.xor(s, r, m)
                    39 -> e.nor(s, r, m)
                    42 -> e.slt(s, r, m)
                    43 -> e.sltu(s, r, m)
                    44 -> e.max(s, r, m)
                    45 -> e.min(s, r, m)
                    46 -> e.msub(s, r, m)
                    47 -> e.msubu(s, r, m)
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
                    0 -> e.bltz(s, r, m)
                    1 -> e.bgez(s, r, m)
                    2 -> e.bltzl(s, r, m)
                    3 -> e.bgezl(s, r, m)
                    16 -> e.bltzal(s, r, m)
                    17 -> e.bgezal(s, r, m)
                    18 -> e.bltzall(s, r, m)
                    19 -> e.bgezall(s, r, m)
                    else -> throw Exception(
                        "Invalid instruction 0x%08X at 0x%08X (3) failed mask 0x%08X".format(
                            i,
                            pc,
                            2031616
                        )
                    )
                }
            2 -> e.j(s, r, m)
            3 -> e.jal(s, r, m)
            4 -> e.beq(s, r, m)
            5 -> e.bne(s, r, m)
            6 -> e.blez(s, r, m)
            7 -> e.bgtz(s, r, m)
            8 -> e.addi(s, r, m)
            9 -> e.addiu(s, r, m)
            10 -> e.slti(s, r, m)
            11 -> e.sltiu(s, r, m)
            12 -> e.andi(s, r, m)
            13 -> e.ori(s, r, m)
            14 -> e.xori(s, r, m)
            15 -> e.lui(s, r, m)
            16 ->
                when (((i shr 0) and 65013759)) {
                    0 -> e.mfc0(s, r, m)
                    4194304 -> e.cfc0(s, r, m)
                    8388608 -> e.mtc0(s, r, m)
                    12582912 -> e.ctc0(s, r, m)
                    33554456 -> e.eret(s, r, m)
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
                    0 -> e.mfc1(s, r, m)
                    2 -> e.cfc1(s, r, m)
                    4 -> e.mtc1(s, r, m)
                    6 -> e.ctc1(s, r, m)
                    8 ->
                        when (((i shr 16) and 64543)) {
                            17408 -> e.bc1f(s, r, m)
                            17409 -> e.bc1t(s, r, m)
                            17410 -> e.bc1fl(s, r, m)
                            17411 -> e.bc1tl(s, r, m)
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
                            1140850688 -> e.add_s(s, r, m)
                            1140850689 -> e.sub_s(s, r, m)
                            1140850690 -> e.mul_s(s, r, m)
                            1140850691 -> e.div_s(s, r, m)
                            1140850692 -> e.sqrt_s(s, r, m)
                            1140850693 -> e.abs_s(s, r, m)
                            1140850694 -> e.mov_s(s, r, m)
                            1140850695 -> e.neg_s(s, r, m)
                            1140850700 -> e.round_w_s(s, r, m)
                            1140850701 -> e.trunc_w_s(s, r, m)
                            1140850702 -> e.ceil_w_s(s, r, m)
                            1140850703 -> e.floor_w_s(s, r, m)
                            1140850724 -> e.cvt_w_s(s, r, m)
                            1140850736 -> e.c_f_s(s, r, m)
                            1140850737 -> e.c_un_s(s, r, m)
                            1140850738 -> e.c_eq_s(s, r, m)
                            1140850739 -> e.c_ueq_s(s, r, m)
                            1140850740 -> e.c_olt_s(s, r, m)
                            1140850741 -> e.c_ult_s(s, r, m)
                            1140850742 -> e.c_ole_s(s, r, m)
                            1140850743 -> e.c_ule_s(s, r, m)
                            1140850744 -> e.c_sf_s(s, r, m)
                            1140850745 -> e.c_ngle_s(s, r, m)
                            1140850746 -> e.c_seq_s(s, r, m)
                            1140850747 -> e.c_ngl_s(s, r, m)
                            1140850748 -> e.c_lt_s(s, r, m)
                            1140850749 -> e.c_nge_s(s, r, m)
                            1140850750 -> e.c_le_s(s, r, m)
                            1140850751 -> e.c_ngt_s(s, r, m)
                            else -> throw Exception(
                                "Invalid instruction 0x%08X at 0x%08X (6) failed mask 0x%08X".format(
                                    i,
                                    pc,
                                    -67108801
                                )
                            )
                        }
                    20 -> e.cvt_s_w(s, r, m)
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
                            9437184 -> e.mfv(s, r, m)
                            9437185 -> e.mfvc(s, r, m)
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
                            9437184 -> e.mtv(s, r, m)
                            9437185 -> e.mtvc(s, r, m)
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
                            18432 -> e.bvf(s, r, m)
                            18433 -> e.bvt(s, r, m)
                            18434 -> e.bvfl(s, r, m)
                            18435 -> e.bvtl(s, r, m)
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
            20 -> e.beql(s, r, m)
            21 -> e.bnel(s, r, m)
            22 -> e.blezl(s, r, m)
            23 -> e.bgtzl(s, r, m)
            24 ->
                when (((i shr 23) and 7)) {
                    0 -> e.vadd(s, r, m)
                    1 -> e.vsub(s, r, m)
                    2 -> e.vsbn(s, r, m)
                    7 -> e.vdiv(s, r, m)
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
                    0 -> e.vmul(s, r, m)
                    1 -> e.vdot(s, r, m)
                    2 -> e.vscl(s, r, m)
                    4 -> e.vhdp(s, r, m)
                    5 -> e.vcrs_t(s, r, m)
                    6 -> e.vdet(s, r, m)
                    else -> throw Exception(
                        "Invalid instruction 0x%08X at 0x%08X (13) failed mask 0x%08X".format(
                            i,
                            pc,
                            58720256
                        )
                    )
                }
            26 -> e.mfvme(s, r, m)
            27 ->
                when (((i shr 23) and 7)) {
                    0 -> e.vcmp(s, r, m)
                    2 -> e.vmin(s, r, m)
                    3 -> e.vmax(s, r, m)
                    5 -> e.vscmp(s, r, m)
                    6 -> e.vsge(s, r, m)
                    7 -> e.vslt(s, r, m)
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
                    0 -> e.halt(s, r, m)
                    36 -> e.mfic(s, r, m)
                    38 -> e.mtic(s, r, m)
                    61 ->
                        when (((i shr 21) and 2047)) {
                            896 -> e.mfdr(s, r, m)
                            900 -> e.mtdr(s, r, m)
                            else -> throw Exception(
                                "Invalid instruction 0x%08X at 0x%08X (15) failed mask 0x%08X".format(
                                    i,
                                    pc,
                                    -2097152
                                )
                            )
                        }
                    62 -> e.dret(s, r, m)
                    63 -> e.dbreak(s, r, m)
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
                    0 -> e.ext(s, r, m)
                    4 -> e.ins(s, r, m)
                    32 ->
                        when (((i shr 6) and 67076127)) {
                            32505858 -> e.wsbh(s, r, m)
                            32505859 -> e.wsbw(s, r, m)
                            32505872 -> e.seb(s, r, m)
                            32505876 -> e.bitrev(s, r, m)
                            32505880 -> e.seh(s, r, m)
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
            32 -> e.lb(s, r, m)
            33 -> e.lh(s, r, m)
            34 -> e.lwl(s, r, m)
            35 -> e.lw(s, r, m)
            36 -> e.lbu(s, r, m)
            37 -> e.lhu(s, r, m)
            38 -> e.lwr(s, r, m)
            40 -> e.sb(s, r, m)
            41 -> e.sh(s, r, m)
            42 -> e.swl(s, r, m)
            43 -> e.sw(s, r, m)
            44 -> e.mtvme(s, r, m)
            46 -> e.swr(s, r, m)
            47 -> e.cache(s, r, m)
            48 -> e.ll(s, r, m)
            49 -> e.lwc1(s, r, m)
            50 -> e.lv_s(s, r, m)
            52 ->
                when (((i shr 24) and 3)) {
                    0 ->
                        when (((i shr 21) and 2023)) {
                            1664 ->
                                when (((i shr 16) and 799)) {
                                    0 -> e.vmov(s, r, m)
                                    1 -> e.vabs(s, r, m)
                                    2 -> e.vneg(s, r, m)
                                    3 -> e.vidt(s, r, m)
                                    4 -> e.vsat0(s, r, m)
                                    5 -> e.vsat1(s, r, m)
                                    6 -> e.vzero(s, r, m)
                                    7 -> e.vone(s, r, m)
                                    16 -> e.vrcp(s, r, m)
                                    17 -> e.vrsq(s, r, m)
                                    18 -> e.vsin(s, r, m)
                                    19 -> e.vcos(s, r, m)
                                    20 -> e.vexp2(s, r, m)
                                    21 -> e.vlog2(s, r, m)
                                    22 -> e.vsqrt(s, r, m)
                                    23 -> e.vasin(s, r, m)
                                    24 -> e.vnrcp(s, r, m)
                                    26 -> e.vnsin(s, r, m)
                                    28 -> e.vrexp2(s, r, m)
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
                                    0 -> e.vrnds(s, r, m)
                                    1 -> e.vrndi(s, r, m)
                                    2 -> e.vrndf1(s, r, m)
                                    3 -> e.vrndf2(s, r, m)
                                    18 -> e.vf2h(s, r, m)
                                    19 -> e.vh2f(s, r, m)
                                    22 -> e.vsbz(s, r, m)
                                    23 -> e.vlgb(s, r, m)
                                    24 -> e.vuc2i(s, r, m)
                                    25 -> e.vc2i(s, r, m)
                                    26 -> e.vus2i(s, r, m)
                                    27 -> e.vs2i(s, r, m)
                                    28 -> e.vi2uc(s, r, m)
                                    29 -> e.vi2c(s, r, m)
                                    30 -> e.vi2us(s, r, m)
                                    31 -> e.vi2s(s, r, m)
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
                                    0 -> e.vsrt1(s, r, m)
                                    1 -> e.vsrt2(s, r, m)
                                    2 -> e.vbfy1(s, r, m)
                                    3 -> e.vbfy2(s, r, m)
                                    4 -> e.vocp(s, r, m)
                                    5 -> e.vsocp(s, r, m)
                                    6 -> e.vfad(s, r, m)
                                    7 -> e.vavg(s, r, m)
                                    8 -> e.vsrt3(s, r, m)
                                    9 -> e.vsrt4(s, r, m)
                                    10 -> e.vsgn(s, r, m)
                                    16 -> e.vmfvc(s, r, m)
                                    17 -> e.vmtvc(s, r, m)
                                    25 -> e.vt4444_q(s, r, m)
                                    26 -> e.vt5551_q(s, r, m)
                                    27 -> e.vt5650_q(s, r, m)
                                    else -> throw Exception(
                                        "Invalid instruction 0x%08X at 0x%08X (21) failed mask 0x%08X".format(
                                            i,
                                            pc,
                                            52363264
                                        )
                                    )
                                }
                            1667 -> e.vcst(s, r, m)
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
                            1664 -> e.vf2in(s, r, m)
                            1665 -> e.vf2iz(s, r, m)
                            1666 -> e.vf2iu(s, r, m)
                            1667 -> e.vf2id(s, r, m)
                            1668 -> e.vi2f(s, r, m)
                            1669 ->
                                when (((i shr 19) and 99)) {
                                    64 -> e.vcmovt(s, r, m)
                                    65 -> e.vcmovf(s, r, m)
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
                    3 -> e.vwbn(s, r, m)
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
                    0 -> e.lvl_q(s, r, m)
                    1 -> e.lvr_q(s, r, m)
                    else -> throw Exception(
                        "Invalid instruction 0x%08X at 0x%08X (26) failed mask 0x%08X".format(
                            i,
                            pc,
                            2
                        )
                    )
                }
            54 -> e.lv_q(s, r, m)
            55 ->
                when (((i shr 24) and 3)) {
                    0 -> e.vpfxs(s, r, m)
                    1 -> e.vpfxt(s, r, m)
                    2 -> e.vpfxd(s, r, m)
                    3 ->
                        when (((i shr 23) and 505)) {
                            440 -> e.viim(s, r, m)
                            441 -> e.vfim(s, r, m)
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
            56 -> e.sc(s, r, m)
            57 -> e.swc1(s, r, m)
            58 -> e.sv_s(s, r, m)
            60 ->
                when (((i shr 23) and 7)) {
                    0 -> e.vmmul(s, r, m)
                    1 ->
                        when (((i shr 7) and 33030401)) {
                            31457280 -> e.vhtfm2(s, r, m)
                            31457281 -> e.vtfm2(s, r, m)
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
                            31457281 -> e.vhtfm3(s, r, m)
                            31457536 -> e.vtfm3(s, r, m)
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
                            31457536 -> e.vhtfm4(s, r, m)
                            31457537 -> e.vtfm4(s, r, m)
                            else -> throw Exception(
                                "Invalid instruction 0x%08X at 0x%08X (31) failed mask 0x%08X".format(
                                    i,
                                    pc,
                                    -67075968
                                )
                            )
                        }
                    4 -> e.vmscl(s, r, m)
                    5 ->
                        when (((i shr 7) and 33030401)) {
                            31457536 -> e.vcrsp_t(s, r, m)
                            31457537 -> e.vqmul(s, r, m)
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
                                    896 -> e.vmmov(s, r, m)
                                    899 -> e.vmidt(s, r, m)
                                    902 -> e.vmzero(s, r, m)
                                    903 -> e.vmone(s, r, m)
                                    else -> throw Exception(
                                        "Invalid instruction 0x%08X at 0x%08X (33) failed mask 0x%08X".format(
                                            i,
                                            pc,
                                            60751872
                                        )
                                    )
                                }
                            1921 -> e.vrot(s, r, m)
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
                    0 -> e.svl_q(s, r, m)
                    1 -> e.svr_q(s, r, m)
                    else -> throw Exception(
                        "Invalid instruction 0x%08X at 0x%08X (36) failed mask 0x%08X".format(
                            i,
                            pc,
                            2
                        )
                    )
                }
            62 -> e.sv_q(s, r, m)
            63 ->
                when (((i shr 0) and 67108863)) {
                    67043328 -> e.vnop(s, r, m)
                    67044128 -> e.vsync(s, r, m)
                    67044365 -> e.vflush(s, r, m)
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
