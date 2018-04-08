package com.soywiz.kpspemu.cpu

import com.soywiz.korio.lang.*

object Instructions {
    val add = ID("add", VM("000000:rs:rt:rd:00000:100000"), "%d, %s, %t", ADDR_TYPE_NONE, 0)
    val addu = ID("addu", VM("000000:rs:rt:rd:00000:100001"), "%d, %s, %t", ADDR_TYPE_NONE, 0)
    val addi = ID("addi", VM("001000:rs:rt:imm16"), "%t, %s, %i", ADDR_TYPE_NONE, 0)
    val addiu = ID("addiu", VM("001001:rs:rt:imm16"), "%t, %s, %i", ADDR_TYPE_NONE, 0)
    val sub = ID("sub", VM("000000:rs:rt:rd:00000:100010"), "%d, %s, %t", ADDR_TYPE_NONE, 0)
    val subu = ID("subu", VM("000000:rs:rt:rd:00000:100011"), "%d, %s, %t", ADDR_TYPE_NONE, 0)
    val and = ID("and", VM("000000:rs:rt:rd:00000:100100"), "%d, %s, %t", ADDR_TYPE_NONE, 0)
    val andi = ID("andi", VM("001100:rs:rt:imm16"), "%t, %s, %I", ADDR_TYPE_NONE, 0)
    val nor = ID("nor", VM("000000:rs:rt:rd:00000:100111"), "%d, %s, %t", ADDR_TYPE_NONE, 0)
    val or = ID("or", VM("000000:rs:rt:rd:00000:100101"), "%d, %s, %t", ADDR_TYPE_NONE, 0)
    val ori = ID("ori", VM("001101:rs:rt:imm16"), "%t, %s, %I", ADDR_TYPE_NONE, 0)
    val xor = ID("xor", VM("000000:rs:rt:rd:00000:100110"), "%d, %s, %t", ADDR_TYPE_NONE, 0)
    val xori = ID("xori", VM("001110:rs:rt:imm16"), "%t, %s, %I", ADDR_TYPE_NONE, 0)
    val sll = ID("sll", VM("000000:00000:rt:rd:sa:000000"), "%d, %t, %a", ADDR_TYPE_NONE, 0)
    val sllv = ID("sllv", VM("000000:rs:rt:rd:00000:000100"), "%d, %t, %s", ADDR_TYPE_NONE, 0)
    val sra = ID("sra", VM("000000:00000:rt:rd:sa:000011"), "%d, %t, %a", ADDR_TYPE_NONE, 0)
    val srav = ID("srav", VM("000000:rs:rt:rd:00000:000111"), "%d, %t, %s", ADDR_TYPE_NONE, 0)
    val srl = ID("srl", VM("000000:00000:rt:rd:sa:000010"), "%d, %t, %a", ADDR_TYPE_NONE, 0)
    val srlv = ID("srlv", VM("000000:rs:rt:rd:00000:000110"), "%d, %t, %s", ADDR_TYPE_NONE, 0)
    val rotr = ID("rotr", VM("000000:00001:rt:rd:sa:000010"), "%d, %t, %a", ADDR_TYPE_NONE, 0)
    val rotrv = ID("rotrv", VM("000000:rs:rt:rd:00001:000110"), "%d, %t, %s", ADDR_TYPE_NONE, 0)
    val slt = ID("slt", VM("000000:rs:rt:rd:00000:101010"), "%d, %s, %t", ADDR_TYPE_NONE, 0)
    val slti = ID("slti", VM("001010:rs:rt:imm16"), "%t, %s, %i", ADDR_TYPE_NONE, 0)
    val sltu = ID("sltu", VM("000000:rs:rt:rd:00000:101011"), "%d, %s, %t", ADDR_TYPE_NONE, 0)
    val sltiu = ID("sltiu", VM("001011:rs:rt:imm16"), "%t, %s, %i", ADDR_TYPE_NONE, 0)
    val lui = ID("lui", VM("001111:00000:rt:imm16"), "%t, %I", ADDR_TYPE_NONE, 0)
    val seb = ID("seb", VM("011111:00000:rt:rd:10000:100000"), "%d, %t", ADDR_TYPE_NONE, 0)
    val seh = ID("seh", VM("011111:00000:rt:rd:11000:100000"), "%d, %t", ADDR_TYPE_NONE, 0)
    val bitrev = ID("bitrev", VM("011111:00000:rt:rd:10100:100000"), "%d, %t", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val max = ID("max", VM("000000:rs:rt:rd:00000:101100"), "%d, %s, %t", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val min = ID("min", VM("000000:rs:rt:rd:00000:101101"), "%d, %s, %t", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val div = ID("div", VM("000000:rs:rt:00000:00000:011010"), "%s, %t", ADDR_TYPE_NONE, 0)
    val divu = ID("divu", VM("000000:rs:rt:00000:00000:011011"), "%s, %t", ADDR_TYPE_NONE, 0)
    val mult = ID("mult", VM("000000:rs:rt:00000:00000:011000"), "%s, %t", ADDR_TYPE_NONE, 0)
    val multu = ID("multu", VM("000000:rs:rt:00000:00000:011001"), "%s, %t", ADDR_TYPE_NONE, 0)
    val madd = ID("madd", VM("000000:rs:rt:00000:00000:011100"), "%s, %t", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val maddu = ID("maddu", VM("000000:rs:rt:00000:00000:011101"), "%s, %t", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val msub = ID("msub", VM("000000:rs:rt:00000:00000:101110"), "%s, %t", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val msubu = ID("msubu", VM("000000:rs:rt:00000:00000:101111"), "%s, %t", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val mfhi = ID("mfhi", VM("000000:00000:00000:rd:00000:010000"), "%d", ADDR_TYPE_NONE, 0)
    val mflo = ID("mflo", VM("000000:00000:00000:rd:00000:010010"), "%d", ADDR_TYPE_NONE, 0)
    val mthi = ID("mthi", VM("000000:rs:00000:00000:00000:010001"), "%s", ADDR_TYPE_NONE, 0)
    val mtlo = ID("mtlo", VM("000000:rs:00000:00000:00000:010011"), "%s", ADDR_TYPE_NONE, 0)
    val movz = ID("movz", VM("000000:rs:rt:rd:00000:001010"), "%d, %s, %t", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val movn = ID("movn", VM("000000:rs:rt:rd:00000:001011"), "%d, %s, %t", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val ext = ID("ext", VM("011111:rs:rt:msb:lsb:000000"), "%t, %s, %a, %ne", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val ins = ID("ins", VM("011111:rs:rt:msb:lsb:000100"), "%t, %s, %a, %ni", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val clz = ID("clz", VM("000000:rs:00000:rd:00000:010110"), "%d, %s", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val clo = ID("clo", VM("000000:rs:00000:rd:00000:010111"), "%d, %s", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val wsbh = ID("wsbh", VM("011111:00000:rt:rd:00010:100000"), "%d, %t", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val wsbw = ID("wsbw", VM("011111:00000:rt:rd:00011:100000"), "%d, %t", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val beq = ID("beq", VM("000100:rs:rt:imm16"), "%s, %t, %O", ADDR_TYPE_16, INSTR_TYPE_B)
    val beql = ID("beql", VM("010100:rs:rt:imm16"), "%s, %t, %O", ADDR_TYPE_16, INSTR_TYPE_B or INSTR_TYPE_LIKELY)
    val bgez = ID("bgez", VM("000001:rs:00001:imm16"), "%s, %O", ADDR_TYPE_16, INSTR_TYPE_B)
    val bgezl = ID("bgezl", VM("000001:rs:00011:imm16"), "%s, %O", ADDR_TYPE_16, INSTR_TYPE_B or INSTR_TYPE_LIKELY)
    val bgezal = ID("bgezal", VM("000001:rs:10001:imm16"), "%s, %O", ADDR_TYPE_16, INSTR_TYPE_JAL)
    val bgezall =
        ID("bgezall", VM("000001:rs:10011:imm16"), "%s, %O", ADDR_TYPE_16, INSTR_TYPE_LIKELY or INSTR_TYPE_JAL)
    val bltz = ID("bltz", VM("000001:rs:00000:imm16"), "%s, %O", ADDR_TYPE_16, INSTR_TYPE_B)
    val bltzl = ID("bltzl", VM("000001:rs:00010:imm16"), "%s, %O", ADDR_TYPE_16, INSTR_TYPE_B or INSTR_TYPE_LIKELY)
    val bltzal = ID("bltzal", VM("000001:rs:10000:imm16"), "%s, %O", ADDR_TYPE_16, INSTR_TYPE_JAL)
    val bltzall =
        ID("bltzall", VM("000001:rs:10010:imm16"), "%s, %O", ADDR_TYPE_16, INSTR_TYPE_LIKELY or INSTR_TYPE_JAL)
    val blez = ID("blez", VM("000110:rs:00000:imm16"), "%s, %O", ADDR_TYPE_16, INSTR_TYPE_B)
    val blezl = ID("blezl", VM("010110:rs:00000:imm16"), "%s, %O", ADDR_TYPE_16, INSTR_TYPE_B or INSTR_TYPE_LIKELY)
    val bgtz = ID("bgtz", VM("000111:rs:00000:imm16"), "%s, %O", ADDR_TYPE_16, INSTR_TYPE_B)
    val bgtzl = ID("bgtzl", VM("010111:rs:00000:imm16"), "%s, %O", ADDR_TYPE_16, INSTR_TYPE_B or INSTR_TYPE_LIKELY)
    val bne = ID("bne", VM("000101:rs:rt:imm16"), "%s, %t, %O", ADDR_TYPE_16, INSTR_TYPE_B)
    val bnel = ID("bnel", VM("010101:rs:rt:imm16"), "%s, %t, %O", ADDR_TYPE_16, INSTR_TYPE_B or INSTR_TYPE_LIKELY)
    val j = ID("j", VM("000010:imm26"), "%j", ADDR_TYPE_26, INSTR_TYPE_JUMP)
    val jr = ID("jr", VM("000000:rs:00000:00000:00000:001000"), "%J", ADDR_TYPE_REG, INSTR_TYPE_JUMP)
    val jalr = ID("jalr", VM("000000:rs:00000:rd:00000:001001"), "%J, %d", ADDR_TYPE_REG, INSTR_TYPE_JAL)
    val jal = ID("jal", VM("000011:imm26"), "%j", ADDR_TYPE_26, INSTR_TYPE_JAL)
    val bc1f = ID("bc1f", VM("010001:01000:00000:imm16"), "%O", ADDR_TYPE_16, INSTR_TYPE_B)
    val bc1t = ID("bc1t", VM("010001:01000:00001:imm16"), "%O", ADDR_TYPE_16, INSTR_TYPE_B)
    val bc1fl = ID("bc1fl", VM("010001:01000:00010:imm16"), "%O", ADDR_TYPE_16, INSTR_TYPE_B or INSTR_TYPE_LIKELY)
    val bc1tl = ID("bc1tl", VM("010001:01000:00011:imm16"), "%O", ADDR_TYPE_16, INSTR_TYPE_B or INSTR_TYPE_LIKELY)
    val lb = ID("lb", VM("100000:rs:rt:imm16"), "%t, %i(%s)", ADDR_TYPE_NONE, 0)
    val lh = ID("lh", VM("100001:rs:rt:imm16"), "%t, %i(%s)", ADDR_TYPE_NONE, 0)
    val lw = ID("lw", VM("100011:rs:rt:imm16"), "%t, %i(%s)", ADDR_TYPE_NONE, 0)
    val lwl = ID("lwl", VM("100010:rs:rt:imm16"), "%t, %i(%s)", ADDR_TYPE_NONE, 0)
    val lwr = ID("lwr", VM("100110:rs:rt:imm16"), "%t, %i(%s)", ADDR_TYPE_NONE, 0)
    val lbu = ID("lbu", VM("100100:rs:rt:imm16"), "%t, %i(%s)", ADDR_TYPE_NONE, 0)
    val lhu = ID("lhu", VM("100101:rs:rt:imm16"), "%t, %i(%s)", ADDR_TYPE_NONE, 0)
    val sb = ID("sb", VM("101000:rs:rt:imm16"), "%t, %i(%s)", ADDR_TYPE_NONE, 0)
    val sh = ID("sh", VM("101001:rs:rt:imm16"), "%t, %i(%s)", ADDR_TYPE_NONE, 0)
    val sw = ID("sw", VM("101011:rs:rt:imm16"), "%t, %i(%s)", ADDR_TYPE_NONE, 0)
    val swl = ID("swl", VM("101010:rs:rt:imm16"), "%t, %i(%s)", ADDR_TYPE_NONE, 0)
    val swr = ID("swr", VM("101110:rs:rt:imm16"), "%t, %i(%s)", ADDR_TYPE_NONE, 0)
    val ll = ID("ll", VM("110000:rs:rt:imm16"), "%t, %O", ADDR_TYPE_NONE, 0)
    val sc = ID("sc", VM("111000:rs:rt:imm16"), "%t, %O", ADDR_TYPE_NONE, 0)
    val lwc1 = ID("lwc1", VM("110001:rs:ft:imm16"), "%T, %i(%s)", ADDR_TYPE_NONE, 0)
    val swc1 = ID("swc1", VM("111001:rs:ft:imm16"), "%T, %i(%s)", ADDR_TYPE_NONE, 0)
    val add_s = ID("add.s", VM("010001:10000:ft:fs:fd:000000"), "%D, %S, %T", ADDR_TYPE_NONE, 0)
    val sub_s = ID("sub.s", VM("010001:10000:ft:fs:fd:000001"), "%D, %S, %T", ADDR_TYPE_NONE, 0)
    val mul_s = ID("mul.s", VM("010001:10000:ft:fs:fd:000010"), "%D, %S, %T", ADDR_TYPE_NONE, 0)
    val div_s = ID("div.s", VM("010001:10000:ft:fs:fd:000011"), "%D, %S, %T", ADDR_TYPE_NONE, 0)
    val sqrt_s = ID("sqrt.s", VM("010001:10000:00000:fs:fd:000100"), "%D, %S", ADDR_TYPE_NONE, 0)
    val abs_s = ID("abs.s", VM("010001:10000:00000:fs:fd:000101"), "%D, %S", ADDR_TYPE_NONE, 0)
    val mov_s = ID("mov.s", VM("010001:10000:00000:fs:fd:000110"), "%D, %S", ADDR_TYPE_NONE, 0)
    val neg_s = ID("neg.s", VM("010001:10000:00000:fs:fd:000111"), "%D, %S", ADDR_TYPE_NONE, 0)
    val round_w_s = ID("round.w.s", VM("010001:10000:00000:fs:fd:001100"), "%D, %S", ADDR_TYPE_NONE, 0)
    val trunc_w_s = ID("trunc.w.s", VM("010001:10000:00000:fs:fd:001101"), "%D, %S", ADDR_TYPE_NONE, 0)
    val ceil_w_s = ID("ceil.w.s", VM("010001:10000:00000:fs:fd:001110"), "%D, %S", ADDR_TYPE_NONE, 0)
    val floor_w_s = ID("floor.w.s", VM("010001:10000:00000:fs:fd:001111"), "%D, %S", ADDR_TYPE_NONE, 0)
    val cvt_s_w = ID("cvt.s.w", VM("010001:10100:00000:fs:fd:100000"), "%D, %S", ADDR_TYPE_NONE, 0)
    val cvt_w_s = ID("cvt.w.s", VM("010001:10000:00000:fs:fd:100100"), "%D, %S", ADDR_TYPE_NONE, 0)
    val mfc1 = ID("mfc1", VM("010001:00000:rt:c1dr:00000:000000"), "%t, %S", ADDR_TYPE_NONE, 0)
    val mtc1 = ID("mtc1", VM("010001:00100:rt:c1dr:00000:000000"), "%t, %S", ADDR_TYPE_NONE, 0)
    val cfc1 = ID("cfc1", VM("010001:00010:rt:c1cr:00000:000000"), "%t, %p", ADDR_TYPE_NONE, 0)
    val ctc1 = ID("ctc1", VM("010001:00110:rt:c1cr:00000:000000"), "%t, %p", ADDR_TYPE_NONE, 0)
    val c_f_s = ID("c.f.s", VM("010001:10000:ft:fs:00000:11:0000"), "%S, %T", ADDR_TYPE_NONE, 0)
    val c_un_s = ID("c.un.s", VM("010001:10000:ft:fs:00000:11:0001"), "%S, %T", ADDR_TYPE_NONE, 0)
    val c_eq_s = ID("c.eq.s", VM("010001:10000:ft:fs:00000:11:0010"), "%S, %T", ADDR_TYPE_NONE, 0)
    val c_ueq_s = ID("c.ueq.s", VM("010001:10000:ft:fs:00000:11:0011"), "%S, %T", ADDR_TYPE_NONE, 0)
    val c_olt_s = ID("c.olt.s", VM("010001:10000:ft:fs:00000:11:0100"), "%S, %T", ADDR_TYPE_NONE, 0)
    val c_ult_s = ID("c.ult.s", VM("010001:10000:ft:fs:00000:11:0101"), "%S, %T", ADDR_TYPE_NONE, 0)
    val c_ole_s = ID("c.ole.s", VM("010001:10000:ft:fs:00000:11:0110"), "%S, %T", ADDR_TYPE_NONE, 0)
    val c_ule_s = ID("c.ule.s", VM("010001:10000:ft:fs:00000:11:0111"), "%S, %T", ADDR_TYPE_NONE, 0)
    val c_sf_s = ID("c.sf.s", VM("010001:10000:ft:fs:00000:11:1000"), "%S, %T", ADDR_TYPE_NONE, 0)
    val c_ngle_s = ID("c.ngle.s", VM("010001:10000:ft:fs:00000:11:1001"), "%S, %T", ADDR_TYPE_NONE, 0)
    val c_seq_s = ID("c.seq.s", VM("010001:10000:ft:fs:00000:11:1010"), "%S, %T", ADDR_TYPE_NONE, 0)
    val c_ngl_s = ID("c.ngl.s", VM("010001:10000:ft:fs:00000:11:1011"), "%S, %T", ADDR_TYPE_NONE, 0)
    val c_lt_s = ID("c.lt.s", VM("010001:10000:ft:fs:00000:11:1100"), "%S, %T", ADDR_TYPE_NONE, 0)
    val c_nge_s = ID("c.nge.s", VM("010001:10000:ft:fs:00000:11:1101"), "%S, %T", ADDR_TYPE_NONE, 0)
    val c_le_s = ID("c.le.s", VM("010001:10000:ft:fs:00000:11:1110"), "%S, %T", ADDR_TYPE_NONE, 0)
    val c_ngt_s = ID("c.ngt.s", VM("010001:10000:ft:fs:00000:11:1111"), "%S, %T", ADDR_TYPE_NONE, 0)
    val syscall = ID("syscall", VM("000000:imm20:001100"), "%C", ADDR_TYPE_NONE, INSTR_TYPE_SYSCALL)
    val cache = ID("cache", VM("101111:rs:-----:imm16"), "%k, %o", ADDR_TYPE_NONE, 0)
    val sync = ID("sync", VM("000000:00000:00000:00000:00000:001111"), "", ADDR_TYPE_NONE, 0)
    val _break = ID("break", VM("000000:imm20:001101"), "%c", ADDR_TYPE_NONE, INSTR_TYPE_BREAK)
    val dbreak = ID(
        "dbreak",
        VM("011100:00000:00000:00000:00000:111111"),
        "",
        ADDR_TYPE_NONE,
        INSTR_TYPE_PSP or INSTR_TYPE_BREAK
    )
    val halt = ID("halt", VM("011100:00000:00000:00000:00000:000000"), "", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val dret = ID("dret", VM("011100:00000:00000:00000:00000:111110"), "", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val eret = ID("eret", VM("010000:10000:00000:00000:00000:011000"), "", ADDR_TYPE_NONE, 0)
    val mfic = ID("mfic", VM("011100:rt:00000:00000:00000:100100"), "%t, %p", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val mtic = ID("mtic", VM("011100:rt:00000:00000:00000:100110"), "%t, %p", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val mfdr = ID("mfdr", VM("011100:00000:----------:00000:111101"), "%t, %r", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val mtdr = ID("mtdr", VM("011100:00100:----------:00000:111101"), "%t, %r", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val cfc0 = ID("cfc0", VM("010000:00010:----------:00000:000000"), "%t, %p", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val ctc0 = ID("ctc0", VM("010000:00110:----------:00000:000000"), "%t, %p", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val mfc0 = ID("mfc0", VM("010000:00000:----------:00000:000000"), "%t, %0", ADDR_TYPE_NONE, 0)
    val mtc0 = ID("mtc0", VM("010000:00100:----------:00000:000000"), "%t, %0", ADDR_TYPE_NONE, 0)
    val mfv = ID("mfv", VM("010010:00:011:rt:0:0000000:0:vd"), "%t, %zs", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val mfvc = ID("mfvc", VM("010010:00:011:rt:0:0000000:1:vd"), "%t, %2d", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val mtv = ID("mtv", VM("010010:00:111:rt:0:0000000:0:vd"), "%t, %zs", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val mtvc = ID("mtvc", VM("010010:00:111:rt:0:0000000:1:vd"), "%t, %2d", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val lv_s = ID("lv.s", VM("110010:rs:vt5:imm14:vt2"), "%Xs, %Y", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val lv_q = ID("lv.q", VM("110110:rs:vt5:imm14:0:vt1"), "%Xq, %Y", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val lvl_q = ID("lvl.q", VM("110101:rs:vt5:imm14:0:vt1"), "%Xq, %Y", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val lvr_q = ID("lvr.q", VM("110101:rs:vt5:imm14:1:vt1"), "%Xq, %Y", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val sv_q = ID("sv.q", VM("111110:rs:vt5:imm14:0:vt1"), "%Xq, %Y", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val vdot = ID("vdot", VM("011001:001:vt:two:vs:one:vd"), "%zs, %yp, %xp", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val vscl = ID("vscl", VM("011001:010:vt:two:vs:one:vd"), "%zp, %yp, %xs", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val vsge = ID("vsge", VM("011011:110:vt:two:vs:one:vd"), "%zp, %yp, %xp", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val vslt = ID("vslt", VM("011011:111:vt:two:vs:one:vd"), "%zp, %yp, %xp", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val vrot = ID("vrot", VM("111100:111:01:imm5:two:vs:one:vd"), "%zp, %ys, %vr", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val vzero = ID("vzero", VM("110100:00:000:0:0110:two:0000000:one:vd"), "%zp", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val vone = ID("vone", VM("110100:00:000:0:0111:two:0000000:one:vd"), "%zp", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val vmov = ID("vmov", VM("110100:00:000:0:0000:two:vs:one:vd"), "%zp, %yp", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val vabs = ID("vabs", VM("110100:00:000:0:0001:two:vs:one:vd"), "%zp, %yp", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val vneg = ID("vneg", VM("110100:00:000:0:0010:two:vs:one:vd"), "%zp, %yp", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val vocp = ID("vocp", VM("110100:00:010:0:0100:two:vs:one:vd"), "%zp, %yp", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val vsgn = ID("vsgn", VM("110100:00:010:0:1010:two:vs:one:vd"), "%zp, %yp", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val vrcp = ID("vrcp", VM("110100:00:000:1:0000:two:vs:one:vd"), "%zp, %yp", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val vrsq = ID("vrsq", VM("110100:00:000:1:0001:two:vs:one:vd"), "%zp, %yp", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val vsin = ID("vsin", VM("110100:00:000:1:0010:two:vs:one:vd"), "%zp, %yp", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val vcos = ID("vcos", VM("110100:00:000:1:0011:two:vs:one:vd"), "%zp, %yp", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val vexp2 = ID("vexp2", VM("110100:00:000:1:0100:two:vs:one:vd"), "%zp, %yp", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val vlog2 = ID("vlog2", VM("110100:00:000:1:0101:two:vs:one:vd"), "%zp, %yp", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val vsqrt = ID("vsqrt", VM("110100:00:000:1:0110:two:vs:one:vd"), "%zp, %yp", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val vasin = ID("vasin", VM("110100:00:000:1:0111:two:vs:one:vd"), "%zp, %yp", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val vnrcp = ID("vnrcp", VM("110100:00:000:1:1000:two:vs:one:vd"), "%zp, %yp", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val vnsin = ID("vnsin", VM("110100:00:000:1:1010:two:vs:one:vd"), "%zp, %yp", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val vrexp2 = ID("vrexp2", VM("110100:00:000:1:1100:two:vs:one:vd"), "%zp, %yp", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val vsat0 = ID("vsat0", VM("110100:00:000:0:0100:two:vs:one:vd"), "%zp, %yp", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val vsat1 = ID("vsat1", VM("110100:00:000:0:0101:two:vs:one:vd"), "%zp, %yp", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val vcst = ID("vcst", VM("110100:00:011:imm5:two:0000000:one:vd"), "%zp, %vk", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val vmmul = ID("vmmul", VM("111100:000:vt:two:vs:one:vd"), "%zm, %tym, %xm", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val vhdp = ID("vhdp", VM("011001:100:vt:two:vs:one:vd"), "%zs, %yp, %xp", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val vcrs_t = ID("vcrs.t", VM("011001:101:vt:1:vs:0:vd"), "%zt, %yt, %xt", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val vcrsp_t = ID("vcrsp.t", VM("111100:101:vt:1:vs:0:vd"), "%zt, %yt, %xt", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val vi2c = ID("vi2c", VM("110100:00:001:11:101:two:vs:one:vd"), "%zs, %yq", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val vi2uc = ID("vi2uc", VM("110100:00:001:11:100:two:vs:one:vd"), "%zq, %yq", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val vtfm2 = ID("vtfm2", VM("111100:001:vt:0:vs:1:vd"), "%zp, %ym, %xp", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val vtfm3 = ID("vtfm3", VM("111100:010:vt:1:vs:0:vd"), "%zt, %yn, %xt", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val vtfm4 = ID("vtfm4", VM("111100:011:vt:1:vs:1:vd"), "%zq, %yo, %xq", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val vhtfm2 = ID("vhtfm2", VM("111100:001:vt:0:vs:0:vd"), "%zp, %ym, %xp", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val vhtfm3 = ID("vhtfm3", VM("111100:010:vt:0:vs:1:vd"), "%zt, %yn, %xt", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val vhtfm4 = ID("vhtfm4", VM("111100:011:vt:1:vs:0:vd"), "%zq, %yo, %xq", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val vsrt3 = ID("vsrt3", VM("110100:00:010:01000:two:vs:one:vd"), "%zq, %yq", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val vfad = ID("vfad", VM("110100:00:010:00110:two:vs:one:vd"), "%zp, %yp", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val vmin = ID("vmin", VM("011011:010:vt:two:vs:one:vd"), "%zp, %yp, %xp", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val vmax = ID("vmax", VM("011011:011:vt:two:vs:one:vd"), "%zp, %yp, %xp", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val vadd = ID("vadd", VM("011000:000:vt:two:vs:one:vd"), "%zp, %yp, %xp", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val vsub = ID("vsub", VM("011000:001:vt:two:vs:one:vd"), "%zp, %yp, %xp", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val vdiv = ID("vdiv", VM("011000:111:vt:two:vs:one:vd"), "%zp, %yp, %xp", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val vmul = ID("vmul", VM("011001:000:vt:two:vs:one:vd"), "%zp, %yp, %xp", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val vidt = ID("vidt", VM("110100:00:000:0:0011:two:0000000:one:vd"), "%zp", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val vmidt = ID("vmidt", VM("111100:111:00:00011:two:0000000:one:vd"), "%zm", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val viim = ID("viim", VM("110111:11:0:vd:imm16"), "%xs, %vi", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val vmmov = ID("vmmov", VM("111100:111:00:00000:two:vs:one:vd"), "%zm, %ym", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val vmzero = ID("vmzero", VM("111100:111:00:00110:two:0000000:one:vd"), "%zm", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val vmone = ID("vmone", VM("111100:111:00:00111:two:0000000:one:vd"), "%zp", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val vnop = ID("vnop", VM("111111:1111111111:00000:00000000000"), "", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val vsync = ID("vsync", VM("111111:1111111111:00000:01100100000"), "", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val vflush = ID("vflush", VM("111111:1111111111:00000:10000001101"), "", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val vpfxd = ID(
        "vpfxd",
        VM("110111:10:------------:mskw:mskz:msky:mskx:satw:satz:saty:satx"),
        "[%vp4, %vp5, %vp6, %vp7]",
        ADDR_TYPE_NONE,
        INSTR_TYPE_PSP
    )
    val vpfxs = ID(
        "vpfxs",
        VM("110111:00:----:negw:negz:negy:negx:cstw:cstz:csty:cstx:absw:absz:absy:absx:swzw:swzz:swzy:swzx"),
        "[%vp0, %vp1, %vp2, %vp3]",
        ADDR_TYPE_NONE,
        INSTR_TYPE_PSP
    )
    val vpfxt = ID(
        "vpfxt",
        VM("110111:01:----:negw:negz:negy:negx:cstw:cstz:csty:cstx:absw:absz:absy:absx:swzw:swzz:swzy:swzx"),
        "[%vp0, %vp1, %vp2, %vp3]",
        ADDR_TYPE_NONE,
        INSTR_TYPE_PSP
    )
    val vdet = ID("vdet", VM("011001:110:vt:two:vs:one:vd"), "%zs, %yp, %xp", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val vrnds = ID("vrnds", VM("110100:00:001:00:000:two:vs:one:0000000"), "%ys", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val vrndi = ID("vrndi", VM("110100:00:001:00:001:two:0000000:one:vd"), "%zp", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val vrndf1 = ID("vrndf1", VM("110100:00:001:00:010:two:0000000:one:vd"), "%zp", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val vrndf2 = ID("vrndf2", VM("110100:00:001:00:011:two:0000000:one:vd"), "%zp", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val vcmp = ID("vcmp", VM("011011:000:vt:two:vs:one:000:imm4"), "%Zn, %yp, %xp", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val vcmovf =
        ID("vcmovf", VM("110100:10:101:01:imm3:two:vs:one:vd"), "%zp, %yp, %v3", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val vcmovt =
        ID("vcmovt", VM("110100:10:101:00:imm3:two:vs:one:vd"), "%zp, %yp, %v3", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val vavg = ID("vavg", VM("110100:00:010:00111:two:vs:one:vd"), "%zp, %yp", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val vf2id = ID("vf2id", VM("110100:10:011:imm5:two:vs:one:vd"), "%zp, %yp, %v5", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val vf2in = ID("vf2in", VM("110100:10:000:imm5:two:vs:one:vd"), "%zp, %yp, %v5", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val vf2iu = ID("vf2iu", VM("110100:10:010:imm5:two:vs:one:vd"), "%zp, %yp, %v5", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val vf2iz = ID("vf2iz", VM("110100:10:001:imm5:two:vs:one:vd"), "%zp, %yp, %v5", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val vi2f = ID("vi2f", VM("110100:10:100:imm5:two:vs:one:vd"), "%zp, %yp, %v5", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val vscmp = ID("vscmp", VM("011011:101:vt:two:vs:one:vd"), "%zp, %yp, %xp", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val vmscl = ID("vmscl", VM("111100:100:vt:two:vs:one:vd"), "%zm, %ym, %xs", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val vt4444_q = ID("vt4444.q", VM("110100:00:010:11001:two:vs:one:vd"), "%zq, %yq", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val vt5551_q = ID("vt5551.q", VM("110100:00:010:11010:two:vs:one:vd"), "%zq, %yq", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val vt5650_q = ID("vt5650.q", VM("110100:00:010:11011:two:vs:one:vd"), "%zq, %yq", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val vmfvc = ID("vmfvc", VM("110100:00:010:10000:1:imm7:0:vd"), "%zs, %2s", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val vmtvc = ID("vmtvc", VM("110100:00:010:10001:0:vs:1:imm7"), "%2d, %ys", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val mfvme = ID("mfvme", VM("011010--------------------------"), "%t, %i", ADDR_TYPE_NONE, 0)
    val mtvme = ID("mtvme", VM("101100--------------------------"), "%t, %i", ADDR_TYPE_NONE, 0)
    val sv_s = ID("sv.s", VM("111010:rs:vt5:imm14:vt2"), "%Xs, %Y", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val vfim = ID("vfim", VM("110111:11:1:vt:imm16"), "%xs, %vh", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val svl_q = ID("svl.q", VM("111101:rs:vt5:imm14:0:vt1"), "%Xq, %Y", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val svr_q = ID("svr.q", VM("111101:rs:vt5:imm14:1:vt1"), "%Xq, %Y", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val vbfy1 = ID("vbfy1", VM("110100:00:010:00010:two:vs:one:vd"), "%zp, %yp", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val vbfy2 = ID("vbfy2", VM("110100:00:010:00011:two:vs:one:vd"), "%zq, %yq", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val vf2h = ID("vf2h", VM("110100:00:001:10:010:two:vs:one:vd"), "%zs, %yp", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val vh2f = ID("vh2f", VM("110100:00:001:10:011:two:vs:one:vd"), "%zq, %yp", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val vi2s = ID("vi2s", VM("110100:00:001:11:111:two:vs:one:vd"), "%zs, %yp", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val vi2us = ID("vi2us", VM("110100:00:001:11:110:two:vs:one:vd"), "%zq, %yq", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val vlgb = ID("vlgb", VM("110100:00:001:10:111:two:vs:one:vd"), "%zs, %ys", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val vqmul = ID("vqmul", VM("111100:101:vt:1:vs:1:vd"), "%zq, %yq, %xq", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val vs2i = ID("vs2i", VM("110100:00:001:11:011:two:vs:one:vd"), "%zq, %yp", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val vc2i = ID("vc2i", VM("110100:00:001:11:001:two:vs:one:vd"), "%zs, %ys, %xs", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val vuc2i = ID("vuc2i", VM("110100:00:001:11:000:two:vs:one:vd"), "%zq, %yp", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val vsbn = ID("vsbn", VM("011000:010:vt:two:vs:one:vd"), "%zs, %ys, %xs", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val vsbz = ID("vsbz", VM("110100:00:001:10110:two:vs:one:vd"), "%zs, %ys", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val vsocp = ID("vsocp", VM("110100:00:010:00101:two:vs:one:vd"), "%zq, %yp", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val vsrt1 = ID("vsrt1", VM("110100:00:010:00000:two:vs:one:vd"), "%zq, %yq", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val vsrt2 = ID("vsrt2", VM("110100:00:010:00001:two:vs:one:vd"), "%zq, %yq", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val vsrt4 = ID("vsrt4", VM("110100:00:010:01001:two:vs:one:vd"), "%zq, %yq", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val vus2i = ID("vus2i", VM("110100:00:001:11010:two:vs:one:vd"), "%zq, %yp", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val vwbn = ID("vwbn", VM("110100:11:imm8:two:vs:one:vd"), "%zs, %xs, %I", ADDR_TYPE_NONE, INSTR_TYPE_PSP)
    val bvf = ID("bvf", VM("010010:01:000:imm3:00:imm16"), "%Zc, %O", ADDR_TYPE_16, INSTR_TYPE_PSP or INSTR_TYPE_B)
    val bvt = ID("bvt", VM("010010:01:000:imm3:01:imm16"), "%Zc, %O", ADDR_TYPE_16, INSTR_TYPE_PSP or INSTR_TYPE_B)
    val bvfl = ID(
        "bvfl",
        VM("010010:01:000:imm3:10:imm16"),
        "%Zc, %O",
        ADDR_TYPE_16,
        INSTR_TYPE_PSP or INSTR_TYPE_B or INSTR_TYPE_LIKELY
    )
    val bvtl = ID(
        "bvtl",
        VM("010010:01:000:imm3:11:imm16"),
        "%Zc, %O",
        ADDR_TYPE_16,
        INSTR_TYPE_PSP or INSTR_TYPE_B or INSTR_TYPE_LIKELY
    )
    val instructions = listOf(
        add,
        addu,
        addi,
        addiu,
        sub,
        subu,
        and,
        andi,
        nor,
        or,
        ori,
        xor,
        xori,
        sll,
        sllv,
        sra,
        srav,
        srl,
        srlv,
        rotr,
        rotrv,
        slt,
        slti,
        sltu,
        sltiu,
        lui,
        seb,
        seh,
        bitrev,
        max,
        min,
        div,
        divu,
        mult,
        multu,
        madd,
        maddu,
        msub,
        msubu,
        mfhi,
        mflo,
        mthi,
        mtlo,
        movz,
        movn,
        ext,
        ins,
        clz,
        clo,
        wsbh,
        wsbw,
        beq,
        beql,
        bgez,
        bgezl,
        bgezal,
        bgezall,
        bltz,
        bltzl,
        bltzal,
        bltzall,
        blez,
        blezl,
        bgtz,
        bgtzl,
        bne,
        bnel,
        j,
        jr,
        jalr,
        jal,
        bc1f,
        bc1t,
        bc1fl,
        bc1tl,
        lb,
        lh,
        lw,
        lwl,
        lwr,
        lbu,
        lhu,
        sb,
        sh,
        sw,
        swl,
        swr,
        ll,
        sc,
        lwc1,
        swc1,
        add_s,
        sub_s,
        mul_s,
        div_s,
        sqrt_s,
        abs_s,
        mov_s,
        neg_s,
        round_w_s,
        trunc_w_s,
        ceil_w_s,
        floor_w_s,
        cvt_s_w,
        cvt_w_s,
        mfc1,
        mtc1,
        cfc1,
        ctc1,
        c_f_s,
        c_un_s,
        c_eq_s,
        c_ueq_s,
        c_olt_s,
        c_ult_s,
        c_ole_s,
        c_ule_s,
        c_sf_s,
        c_ngle_s,
        c_seq_s,
        c_ngl_s,
        c_lt_s,
        c_nge_s,
        c_le_s,
        c_ngt_s,
        syscall,
        cache,
        sync,
        _break,
        dbreak,
        halt,
        dret,
        eret,
        mfic,
        mtic,
        mfdr,
        mtdr,
        cfc0,
        ctc0,
        mfc0,
        mtc0,
        mfv,
        mfvc,
        mtv,
        mtvc,
        lv_s,
        lv_q,
        lvl_q,
        lvr_q,
        sv_q,
        vdot,
        vscl,
        vsge,
        vslt,
        vrot,
        vzero,
        vone,
        vmov,
        vabs,
        vneg,
        vocp,
        vsgn,
        vrcp,
        vrsq,
        vsin,
        vcos,
        vexp2,
        vlog2,
        vsqrt,
        vasin,
        vnrcp,
        vnsin,
        vrexp2,
        vsat0,
        vsat1,
        vcst,
        vmmul,
        vhdp,
        vcrs_t,
        vcrsp_t,
        vi2c,
        vi2uc,
        vtfm2,
        vtfm3,
        vtfm4,
        vhtfm2,
        vhtfm3,
        vhtfm4,
        vsrt3,
        vfad,
        vmin,
        vmax,
        vadd,
        vsub,
        vdiv,
        vmul,
        vidt,
        vmidt,
        viim,
        vmmov,
        vmzero,
        vmone,
        vnop,
        vsync,
        vflush,
        vpfxd,
        vpfxs,
        vpfxt,
        vdet,
        vrnds,
        vrndi,
        vrndf1,
        vrndf2,
        vcmp,
        vcmovf,
        vcmovt,
        vavg,
        vf2id,
        vf2in,
        vf2iu,
        vf2iz,
        vi2f,
        vscmp,
        vmscl,
        vt4444_q,
        vt5551_q,
        vt5650_q,
        vmfvc,
        vmtvc,
        mfvme,
        mtvme,
        sv_s,
        vfim,
        svl_q,
        svr_q,
        vbfy1,
        vbfy2,
        vf2h,
        vh2f,
        vi2s,
        vi2us,
        vlgb,
        vqmul,
        vs2i,
        vc2i,
        vuc2i,
        vsbn,
        vsbz,
        vsocp,
        vsrt1,
        vsrt2,
        vsrt4,
        vus2i,
        vwbn,
        bvf,
        bvt,
        bvfl,
        bvtl
    )
    val instructionsByName = instructions.map { it.name to it }.toMap()
}

const val ADDR_TYPE_NONE: Int = 0
const val ADDR_TYPE_REG: Int = 1
const val ADDR_TYPE_16: Int = 2
const val ADDR_TYPE_26: Int = 3

const val INSTR_TYPE_PSP: Int = 1 shl 0
const val INSTR_TYPE_SYSCALL: Int = 1 shl 1
const val INSTR_TYPE_B: Int = 1 shl 2
const val INSTR_TYPE_LIKELY: Int = 1 shl 3
const val INSTR_TYPE_JAL: Int = 1 shl 4
const val INSTR_TYPE_JUMP: Int = 1 shl 5
const val INSTR_TYPE_BREAK: Int = 1 shl 6

private fun VM(format: String) = ValueMask(format)
private fun ID(name: String, vm: ValueMask, format: String, addressType: Int, instructionType: Int) =
    InstructionType(name, vm, format, addressType, instructionType)

data class ValueMask(val format: String) {
    val value: Int
    val mask: Int

    init {
        val counts = mapOf(
            "cstw" to 1, "cstz" to 1, "csty" to 1, "cstx" to 1,
            "absw" to 1, "absz" to 1, "absy" to 1, "absx" to 1,
            "mskw" to 1, "mskz" to 1, "msky" to 1, "mskx" to 1,
            "negw" to 1, "negz" to 1, "negy" to 1, "negx" to 1,
            "one" to 1, "two" to 1, "vt1" to 1,
            "vt2" to 2,
            "satw" to 2, "satz" to 2, "saty" to 2, "satx" to 2,
            "swzw" to 2, "swzz" to 2, "swzy" to 2, "swzx" to 2,
            "imm3" to 3,
            "imm4" to 4,
            "fcond" to 4,
            "c0dr" to 5, "c0cr" to 5, "c1dr" to 5, "c1cr" to 5, "imm5" to 5, "vt5" to 5,
            "rs" to 5, "rd" to 5, "rt" to 5, "sa" to 5, "lsb" to 5, "msb" to 5, "fs" to 5, "fd" to 5, "ft" to 5,
            "vs" to 7, "vt" to 7, "vd" to 7, "imm7" to 7,
            "imm8" to 8,
            "imm14" to 14,
            "imm16" to 16,
            "imm20" to 20,
            "imm26" to 26
        )

        var value: Int = 0
        var mask: Int = 0

        for (item in format.split(':')) {
            // normal chunk
            if (Regex("^[01\\-]+$").matches(item)) {
                for (c in item) {
                    value = value shl 1
                    mask = mask shl 1
                    if (c == '0') {
                        value = value or 0; mask = mask or 1; }
                    if (c == '1') {
                        value = value or 1; mask = mask or 1; }
                    if (c == '-') {
                        value = value or 0; mask = mask or 0; }
                }
            }
            // special chunk
            else {
                val displacement = counts[item] ?: throw Exception("Invalid item '$item'")
                value = value shl displacement
                mask = mask shl displacement
            }
        }

        this.value = value
        this.mask = mask
    }
}

data class InstructionType(
    val name: String,
    val vm: ValueMask,
    val format: String,
    val addressType: Int,
    val instructionType: Int
) {
    val formatEscaped by lazy { Regex.escapeReplacement(format) }
    val replacements by lazy { Regex("%\\w+").findAll(format).map { it.value }.toList() }
    val formatRegex by lazy { Regex(formatEscaped.replace(Regex("%\\w+")) { "([\\-\\+\\w]+)" }.replace(Regex("\\s+")) { "\\s*" }) }

    fun match(i32: Int) = (i32 and this.vm.mask) == (this.vm.value and this.vm.mask)
    private fun isInstructionType(mask: Int) = (this.instructionType and mask) != 0
    val isSyscall get() = this.isInstructionType(INSTR_TYPE_SYSCALL)
    val isBreak get() = this.isInstructionType(INSTR_TYPE_BREAK)
    val isBranch get() = this.isInstructionType(INSTR_TYPE_B)
    val isCall get() = this.isInstructionType(INSTR_TYPE_JAL)
    val isJump get() = this.isInstructionType(INSTR_TYPE_JAL) || this.isInstructionType(INSTR_TYPE_JUMP)
    val isJumpNoLink get() = this.isInstructionType(INSTR_TYPE_JUMP)
    val isJal get() = this.isInstructionType(INSTR_TYPE_JAL)
    val isJumpOrBranch get() = this.isBranch || this.isJump
    val isLikely get() = this.isInstructionType(INSTR_TYPE_LIKELY)
    val isRegister get() = this.addressType == ADDR_TYPE_REG
    val isFixedAddressJump get() = this.isJumpOrBranch && !this.isRegister
    val hasDelayedBranch get() = this.isJumpOrBranch
    override fun toString() =
        "InstructionType('${this.name}', ${"%08X".format(this.vm.value)}, ${"%08X".format(this.vm.mask)})"
}

fun String.kescape() = when (this) {
    "break" -> "_$this"
    else -> this.replace('.', '_')
}
