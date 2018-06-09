@file:Suppress("unused", "MemberVisibilityCanBePrivate")

package com.soywiz.kemu

import com.soywiz.korio.lang.*

interface Z80Controller {
    fun memoryRead(addr: Int): Int = 0
    fun memoryWrite(addr: Int, value: Int): Unit = Unit
    fun halt(cpu: Z80): Unit = TODO("halt")
    fun stop(cpu: Z80): Unit = TODO("stop")
    fun illegal(cpu: Z80, opcode: Int): Unit = TODO("Illegal opcode 0x%02X".format(opcode))
}

fun Z80Controller.memoryRead2(addr: Int): Int = (memoryRead((addr + 1) and 0xFF) shl 8) or memoryRead(addr)
fun Z80Controller.memoryReadHigh(addr: Int): Int = memoryRead(0xFF00 or addr)
fun Z80Controller.memoryWrite2(addr: Int, value: Int): Unit =
    run { memoryWrite(addr + 1, (value shr 8) and 0xFF); memoryWrite(addr + 0, (value shr 0) and 0xFF) }

fun Z80Controller.memoryHighWrite(index: Int, value: Int): Unit = memoryWrite(0xFF00 or index, value)

class Z80(val ctrl: Z80Controller) : Z80Controller by ctrl {
    var A = 0x01
    var B = 0x00
    var C = 0x13
    var D = 0x00
    var E = 0xD8
    var HL = 0x014D
    var SP = 0xFFFE
    var PC = 0x0100

    var H: Int; set(value) = run { HL = (HL and 0x00FF) or ((value and 0xFF) shl 8) }; get() = (HL shr 8) and 0xFF
    var L: Int; set(value) = run { HL = (HL and 0xFF00) or ((value and 0xFF) shl 0) }; get() = (HL shr 0) and 0xFF
    var _HL_: Int; set(value) = run { Z = value }; get() = Z
    var Z: Int; set(value) = memoryWrite(HL, value); get() = memoryRead(HL)
    val _PC_ get() = memoryRead(PC)
    var _SP_: Int set(value) = memoryWrite(SP, value); get() = memoryRead(SP)
    val _SP2_ get() = memoryRead2(SP)
    val _PC2_ get() = memoryRead2(PC)

    // @TODO: Use F instead
    var F: Int
        set(v) {
            FZ = ((v and 0x80) != 0)
            FS = ((v and 0x40) != 0)
            FH = ((v and 0x20) != 0)
            FC = ((v and 0x10) != 0)
        }
        get() = (if (FZ) 0x80 else 0) or (if (FS) 0x40 else 0) or (if (FH) 0x20 else 0) or (if (FC) 0x10 else 0)

    var AF
        set(v) = run { A = (v shr 8) and 0xFF; F = (v shr 0) and 0xFF; }
        get() = ((A and 0xFF) shl 8) or (F and 0xFF)

    var BC
        set(v) = run { B = (v shr 8) and 0xFF; C = (v shr 0) and 0xFF; }
        get() = ((B and 0xFF) shl 8) or (C and 0xFF)
    var DE
        set(v) = run { D = (v shr 8) and 0xFF; E = (v shr 0) and 0xFF; }
        get() = ((D and 0xFF) shl 8) or (E and 0xFF)

    var FZ = true // Zero
    var FS = false // Subtract
    var FH = true // HalfCarry
    var FC = true // Carry

    var CPUTicks = 0

    var usedBootROM = false
    var skipPCIncrement = false

    var interruptsEnabled = 0
    var interruptsRequested = 0
    var IRQEnableDelay = 1
    var IME = false

    fun writeSP(v: Int) = run { SP -= 1; memoryWrite(SP, v) }
    fun writeSP2(v: Int) = run { SP -= 2; memoryWrite2(SP, v) }

    fun readSP(): Int = _SP_.apply { SP = (SP + 1) and 0xFFFF }
    fun readSP2(): Int = _SP2_.apply { SP = (SP + 2) and 0xFFFF }

    fun readPC(): Int = _PC_.apply { PC = (PC + 1) and 0xFFFF }
    fun sreadPC(): Int = readPC() shl 24 shr 24
    fun readPC2(): Int = _PC2_.apply { PC = (PC + 2) and 0xFFFF }

    /////
    fun i() {
        val opcode = _PC_
        PC = (PC + 1) and 0xFFFF
        CPUTicks += TickTableNone[opcode]
        when (opcode) {
            0x00 -> i00();0x01 -> i01();0x02 -> i02();0x03 -> i03();0x04 -> i04();0x05 -> i05();0x06 -> i06();0x07 -> i07()
            0x08 -> i08();0x09 -> i09();0x0A -> i0A();0x0B -> i0B();0x0C -> i0C();0x0D -> i0D();0x0E -> i0E();0x0F -> i0F()
            0x10 -> i10();0x11 -> i11();0x12 -> i12();0x13 -> i13();0x14 -> i14();0x15 -> i15();0x16 -> i16();0x17 -> i17()
            0x18 -> i18();0x19 -> i19();0x1A -> i1A();0x1B -> i1B();0x1C -> i1C();0x1D -> i1D();0x1E -> i1E();0x1F -> i1F()
            0x20 -> i20();0x21 -> i21();0x22 -> i22();0x23 -> i23();0x24 -> i24();0x25 -> i25();0x26 -> i26();0x27 -> i27()
            0x28 -> i28();0x29 -> i29();0x2A -> i2A();0x2B -> i2B();0x2C -> i2C();0x2D -> i2D();0x2E -> i2E();0x2F -> i2F()
            0x30 -> i30();0x31 -> i31();0x32 -> i32();0x33 -> i33();0x34 -> i34();0x35 -> i35();0x36 -> i36();0x37 -> i37()
            0x38 -> i38();0x39 -> i39();0x3A -> i3A();0x3B -> i3B();0x3C -> i3C();0x3D -> i3D();0x3E -> i3E();0x3F -> i3F()
            0x40 -> i40();0x41 -> i41();0x42 -> i42();0x43 -> i43();0x44 -> i44();0x45 -> i45();0x46 -> i46();0x47 -> i47()
            0x48 -> i48();0x49 -> i49();0x4A -> i4A();0x4B -> i4B();0x4C -> i4C();0x4D -> i4D();0x4E -> i4E();0x4F -> i4F()
            0x50 -> i50();0x51 -> i51();0x52 -> i52();0x53 -> i53();0x54 -> i54();0x55 -> i55();0x56 -> i56();0x57 -> i57()
            0x58 -> i58();0x59 -> i59();0x5A -> i5A();0x5B -> i5B();0x5C -> i5C();0x5D -> i5D();0x5E -> i5E();0x5F -> i5F()
            0x60 -> i60();0x61 -> i61();0x62 -> i62();0x63 -> i63();0x64 -> i64();0x65 -> i65();0x66 -> i66();0x67 -> i67()
            0x68 -> i68();0x69 -> i69();0x6A -> i6A();0x6B -> i6B();0x6C -> i6C();0x6D -> i6D();0x6E -> i6E();0x6F -> i6F()
            0x70 -> i70();0x71 -> i71();0x72 -> i72();0x73 -> i73();0x74 -> i74();0x75 -> i75();0x76 -> i76();0x77 -> i77()
            0x78 -> i78();0x79 -> i79();0x7A -> i7A();0x7B -> i7B();0x7C -> i7C();0x7D -> i7D();0x7E -> i7E();0x7F -> i7F()
            0x80 -> i80();0x81 -> i81();0x82 -> i82();0x83 -> i83();0x84 -> i84();0x85 -> i85();0x86 -> i86();0x87 -> i87()
            0x88 -> i88();0x89 -> i89();0x8A -> i8A();0x8B -> i8B();0x8C -> i8C();0x8D -> i8D();0x8E -> i8E();0x8F -> i8F()
            0x90 -> i90();0x91 -> i91();0x92 -> i92();0x93 -> i93();0x94 -> i94();0x95 -> i95();0x96 -> i96();0x97 -> i97()
            0x98 -> i98();0x99 -> i99();0x9A -> i9A();0x9B -> i9B();0x9C -> i9C();0x9D -> i9D();0x9E -> i9E();0x9F -> i9F()
            0xA0 -> iA0();0xA1 -> iA1();0xA2 -> iA2();0xA3 -> iA3();0xA4 -> iA4();0xA5 -> iA5();0xA6 -> iA6();0xA7 -> iA7()
            0xA8 -> iA8();0xA9 -> iA9();0xAA -> iAA();0xAB -> iAB();0xAC -> iAC();0xAD -> iAD();0xAE -> iAE();0xAF -> iAF()
            0xB0 -> iB0();0xB1 -> iB1();0xB2 -> iB2();0xB3 -> iB3();0xB4 -> iB4();0xB5 -> iB5();0xB6 -> iB6();0xB7 -> iB7()
            0xB8 -> iB8();0xB9 -> iB9();0xBA -> iBA();0xBB -> iBB();0xBC -> iBC();0xBD -> iBD();0xBE -> iBE();0xBF -> iBF()
            0xC0 -> iC0();0xC1 -> iC1();0xC2 -> iC2();0xC3 -> iC3();0xC4 -> iC4();0xC5 -> iC5();0xC6 -> iC6();0xC7 -> iC7()
            0xC8 -> iC8();0xC9 -> iC9();0xCA -> iCA();0xCB -> iCB();0xCC -> iCC();0xCD -> iCD();0xCE -> iCE();0xCF -> iCF()
            0xD0 -> iD0();0xD1 -> iD1();0xD2 -> iD2();0xD3 -> iD3();0xD4 -> iD4();0xD5 -> iD5();0xD6 -> iD6();0xD7 -> iD7()
            0xD8 -> iD8();0xD9 -> iD9();0xDA -> iDA();0xDB -> iDB();0xDC -> iDC();0xDD -> iDD();0xDE -> iDE();0xDF -> iDF()
            0xE0 -> iE0();0xE1 -> iE1();0xE2 -> iE2();0xE3 -> iE3();0xE4 -> iE4();0xE5 -> iE5();0xE6 -> iE6();0xE7 -> iE7()
            0xE8 -> iE8();0xE9 -> iE9();0xEA -> iEA();0xEB -> iEB();0xEC -> iEC();0xED -> iED();0xEE -> iEE();0xEF -> iEF()
            0xF0 -> iF0();0xF1 -> iF1();0xF2 -> iF2();0xF3 -> iF3();0xF4 -> iF4();0xF5 -> iF5();0xF6 -> iF6();0xF7 -> iF7()
            0xF8 -> iF8();0xF9 -> iF9();0xFA -> iFA();0xFB -> iFB();0xFC -> iFC();0xFD -> iFD();0xFE -> iFE();0xFF -> iFF()
        }
    }

    fun _INC(v: Int): Int =
        run { val res = (v + 1) and 0xFF; FZ = (v == 0); FH = ((v and 0xF) == 0x0); FS = false; res }

    fun _DEC(v: Int): Int = run { val res = (v - 1) and 0xFF; FZ = (v == 0); FH = ((v and 0xF) == 0xF); FS = true; res }
    fun _DEC2(v: Int): Int = (v - 1) and 0xFFFF
    fun _INC2(v: Int): Int = (v + 1) and 0xFFFF

    fun _ADDHL(v: Int) {
        val ds = HL + v
        FH = ((HL and 0xFFF) > (ds and 0xFFF))
        FC = (ds > 0xFFFF)
        HL = ds and 0xFFFF
        FS = false
    }

    fun _ADD(v: Int) {
        val PA = A
        val ds = A + v
        A = ds and 0xFF
        FH = ((ds and 0xF) < (PA and 0xF)); FC = (ds > 0xFF); FZ = (A == 0); FS = false
    }

    fun _ADC(v: Int) {
        val PA = A
        val ds = A + v + (if (FC) 1 else 0)
        A = ds and 0xFF
        FH = ((PA and 0xF) + (v and 0xF) + (if (FC) 1 else 0) > 0xF); FC = (ds > 0xFF); FZ = (A == 0); FS = false
    }

    fun _SUB(v: Int) {
        val PA = A
        val ds = A - v
        A = ds and 0xFF
        FH = ((PA and 0xF) < (ds and 0xF));FC = (ds < 0); FZ = (ds == 0); FS = true
    }

    fun _SBC(v: Int) {
        val PA = A
        val ds = PA - v - (if (FC) 1 else 0)
        A = ds and 0xFF
        FH = ((PA and 0xF) - (v and 0xF) - (if (FC) 1 else 0) < 0); FC = (ds < 0); FZ = (A == 0); FS = true
    }

    fun _AND(v: Int) = run { A = A and v; FZ = (A == 0); FH = true; FS = false; FC = false }
    fun _XOR(v: Int) = run { A = A xor v; FZ = (A == 0); FS = false; FH = false; FC = false }
    fun _OR(v: Int) = run { A = A or v; FZ = (A == 0); FS = false; FC = false; FH = false }
    fun _CP(v: Int) =
        run { val ds = A - v; FH = ((ds and 0xF) > (A and 0xF)); FC = (ds < 0); FZ = (ds == 0); FS = true }

    fun _JP_IF(cond: Boolean) = run { val nPC = readPC2(); if (cond) PC = nPC }
    fun _JR_IF(cond: Boolean) = run { val nPC = (PC + sreadPC()) and 0xFFFF; if (cond) PC = nPC }
    fun _CALL_IF(cond: Boolean) = run { val nPC = readPC2(); if (cond) run { writeSP2(PC); PC = nPC; CPUTicks += 12 } }
    fun _RET_IF(cond: Boolean) = run { if (cond) run { PC = readSP2(); CPUTicks += 12 } }
    fun _RST(index: Int) = run { writeSP(PC); PC = index }

    fun _RLC(v: Int): Int {
        val res = ((v shl 1) and 0xFF) or (if (FC) 1 else 0)
        FC = (v > 0x7F); FH = false; FS = false; FZ = (res == 0)
        return res
    }

    fun _RRC(v: Int): Int {
        val res = (if (FC) 0x80 else 0) or (v shr 1)
        FC = ((v and 0x01) == 0x01); FH = false; FS = false; FZ = (res == 0)
        return res
    }

    fun _RL(v: Int): Int {
        val res = ((v shl 1) and 0xFF) or (if (FC) 1 else 0)
        FC = (v > 0x7F); FH = false; FS = false; FZ = (res == 0)
        return res
    }

    fun _RR(v: Int): Int {
        val res = (if (FC) 0x80 else 0) or (v shr 1)
        FC = ((v and 0x01) == 0x01); FH = false; FS = false; FZ = (res == 0)
        return v
    }

    fun _SLA(v: Int): Int {
        val res = (v shl 1) and 0xFF
        FC = (v > 0x7F); FH = false; FS = false; FZ = (res == 0)
        return res
    }

    fun _SRA(v: Int): Int {
        val res = (v and 0x80) or (v shr 1)
        FC = ((v and 0x01) == 0x01); FH = false; FS = false; FZ = (res == 0)
        return res
    }

    fun _SWAP(v: Int): Int {
        val res = ((v and 0xF) shl 4) or (v shr 4)
        FZ = (res == 0); FC = false; FH = false; FS = false
        return res
    }

    fun _SRL(v: Int): Int {
        FC = ((v and 0x01) == 0x01)
        val res = v shr 1; FH = false; FS = false; FZ = (res == 0)
        return res
    }

    fun _SET(n: Int, v: Int): Int = v or (1 shl n)
    fun _RES(n: Int, v: Int): Int = v and (1 shl n).inv()
    fun _BIT(n: Int, v: Int) = run { FH = true; FS = false; FZ = ((v and (1 shl n)) == 0) }

    //NOP
    fun i00() = Unit

    fun i01() = run { BC = readPC2() } //LD BC, nn
    fun i02() = run { memoryWrite((B shl 8) or C, A) } //LD (BC), A

    fun i1C() = run { E = _INC(E) } //INC E
    fun i04() = run { B = _INC(B) } //INC B
    fun i0C() = run { C = _INC(C) } //INC C
    fun i14() = run { D = _INC(D) } //INC D
    fun i24() = run { H = _INC(H) } //INC H
    fun i2C() = run { L = _INC(L) } //INC L
    fun i34() = run { Z = _INC(Z) } //INC (HL)
    fun i3C() = run { A = _INC(A) } //INC A

    fun i33() = run { SP = _INC2(SP) } //INC SP
    fun i03() = run { BC = _INC2(BC) } //INC BC
    fun i13() = run { DE = _INC2(DE) } //INC DE
    fun i23() = run { HL = _INC2(HL) } //INC HL

    fun i05() = run { B = _DEC(B) } //DEC B
    fun i0D() = run { C = _DEC(C) } //DEC C
    fun i15() = run { D = _DEC(D) } //DEC D
    fun i1D() = run { E = _DEC(E) } //DEC E
    fun i25() = run { H = _DEC(H) } //DEC H
    fun i35() = run { Z = _DEC(Z) } //DEC (HL)
    fun i3D() = run { A = _DEC(A) } //DEC A
    fun i2D() = run { L = _DEC(L) } //DEC L
    fun i3B() = run { SP = _DEC2(SP) } //DEC SP
    fun i0B() = run { BC = _DEC2(BC) } //DEC BC
    fun i1B() = run { DE = _DEC2(DE) } //DEC DE
    fun i2B() = run { HL = _DEC2(HL) } //DEC HL
    fun i07() =
        run { FC = (A > 0x7F); A = ((A shl 1) and 0xFF) or (A shr 7); FZ = false; FS = false; FH = false } //RLCA

    fun i08() = memoryWrite2(readPC2(), SP) //LD (nn), SP
    fun i0A() = run { A = memoryRead((B shl 8) or C) } //LD A, (BC)
    fun i0F() = run { A = (A shr 1) or ((A and 1) shl 7); FC = (A > 0x7F); FZ = false; FS = false; FH = false } //RRCA
    fun i10() = run { ctrl.stop(this) } //STOP
    fun i11() = run { DE = readPC2() } //LD DE, nn
    fun i12() = run { memoryWrite((D shl 8) or E, A) } //LD (DE), A
    fun i17() = run {
        val OA = A; A = ((A shl 1) and 0xFF) or (if (FC) 1 else 0); FC = (OA > 0x7F); FZ = false; FS = false; FH = false
    } //RLA

    fun i1A() = run { A = memoryRead((D shl 8) or E) } //LD A, (DE)
    fun i1F() = run {
        val OA = A; A = (A shr 1) or (if (FC) 0x80 else 0); FC = ((OA and 1) == 1); FZ = false; FS = false; FH = false
    } //RRA

    //DAA
    fun i27() {
        when {
            !FS -> {
                if (FC || A > 0x99) run { A = (A + 0x60) and 0xFF; FC = true }
                if (FH || (A and 0xF) > 0x9) run { A = (A + 0x06) and 0xFF; FH = false }
            }
            FC && FH -> run { A = (A + 0x9A) and 0xFF; FH = false }
            FC -> A = (A + 0xA0) and 0xFF
            FH -> run { A = (A + 0xFA) and 0xFF; FH = false }
        }
        FZ = (A == 0)
    }

    fun i21() = run { HL = readPC2() } //LD HL, nn
    fun i22() = run { _HL_ = A; HL = (HL + 1) and 0xFFFF } //LDI (HL), A
    fun i2A() = run { A = Z; HL = (HL + 1) and 0xFFFF } //LDI A, (HL)
    fun i2F() = run { A = A xor 0xFF; FS = true; FH = true } //CPL
    fun i31() = run { SP = readPC2() } //LD SP, nn
    fun i37() = run { FC = true; FS = false; FH = false } //SCF
    fun i3A() = run { A = Z; HL = (HL - 1) and 0xFFFF } //LDD A, (HL)
    fun i32() = run { _HL_ = A; HL = (HL - 1) and 0xFFFF } //LDD (HL), A
    fun i3E() = run { A = readPC() } //LD A, n
    fun i3F() = run { FC = !FC; FS = false; FH = false } //CCF

    fun i40() = run { B = B } //LD B, B
    fun i41() = run { B = C } //LD B, C
    fun i42() = run { B = D } //LD B, D
    fun i43() = run { B = E } //LD B, E
    fun i44() = run { B = H } //LD B, H
    fun i45() = run { B = L } //LD B, L
    fun i46() = run { B = Z } //LD B, (HL)
    fun i47() = run { B = A } //LD B, A
    fun i06() = run { B = readPC() } //LD B, n

    fun i48() = run { C = B } //LD C, B
    fun i49() = run { C = C } //LD C, C
    fun i4A() = run { C = D } //LD C, D
    fun i4B() = run { C = E } //LD C, E
    fun i4C() = run { C = H } //LD C, H
    fun i4D() = run { C = L } //LD C, L
    fun i4E() = run { C = Z } //LD C, (HL)
    fun i4F() = run { C = A } //LD C, A
    fun i0E() = run { C = readPC() } //LD C, n

    fun i50() = run { D = B } //LD D, B
    fun i51() = run { D = C } //LD D, C
    fun i52() = run { D = D } //LD D, D
    fun i53() = run { D = E } //LD D, E
    fun i54() = run { D = H } //LD D, H
    fun i55() = run { D = L } //LD D, L
    fun i56() = run { D = Z } //LD D, (HL)
    fun i57() = run { D = A } //LD D, A
    fun i16() = run { D = readPC() } //LD D, n


    fun i58() = run { E = B } //LD E, B
    fun i59() = run { E = C } //LD E, C
    fun i5A() = run { E = D } //LD E, D
    fun i5B() = run { E = E } //LD E, E
    fun i5C() = run { E = H } //LD E, H
    fun i5D() = run { E = L } //LD E, L
    fun i5E() = run { E = Z } //LD E, (HL)
    fun i5F() = run { E = A } //LD E, A
    fun i1E() = run { E = readPC() } //LD E, n

    fun i60() = run { H = B } //LD H, B
    fun i61() = run { H = C } //LD H, C
    fun i62() = run { H = D } //LD H, D
    fun i63() = run { H = E } //LD H, E
    fun i64() = run { H = H } //LD H, H
    fun i65() = run { H = L } //LD H, L
    fun i66() = run { H = Z } //LD H, (HL)
    fun i67() = run { H = A } //LD H, A
    fun i26() = run { H = readPC() } //LD H, n

    fun i68() = run { L = B } //LD L, B
    fun i69() = run { L = C } //LD L, C
    fun i6A() = run { L = D } //LD L, D
    fun i6B() = run { L = E } //LD L, E
    fun i6C() = run { L = H } //LD L, H
    fun i6D() = run { L = L } //LD L, L
    fun i6E() = run { L = Z } //LD L, (HL)
    fun i6F() = run { L = A } //LD L, A
    fun i2E() = run { L = readPC() } //LD L, n

    fun i70() = run { Z = B } //LD (HL), B
    fun i71() = run { Z = C } //LD (HL), C
    fun i72() = run { Z = D } //LD (HL), D
    fun i73() = run { Z = E } //LD (HL), E
    fun i74() = run { Z = H } //LD (HL), H
    fun i75() = run { Z = L } //LD (HL), L
    fun i77() = run { Z = A } //LD (HL), A
    fun i36() = run { Z = readPC() } //LD (HL), n
    fun i76() = ctrl.halt(this) //HALT
    fun i78() = run { A = B } //LD A, B
    fun i79() = run { A = C } //LD A, C
    fun i7A() = run { A = D } //LD A, D
    fun i7B() = run { A = E } //LD A, E
    fun i7C() = run { A = H } //LD A, H
    fun i7D() = run { A = L } //LD A, L
    fun i7E() = run { A = Z } //LD, A, (HL)
    fun i7F() = run { A = A } //LD A, A

    fun i80() = _ADD(B) //ADD A, B
    fun i81() = _ADD(C) //ADD A, C
    fun i82() = _ADD(D) //ADD A, D
    fun i83() = _ADD(E) //ADD A, E
    fun i84() = _ADD(H) //ADD A, H
    fun i85() = _ADD(L) //ADD A, L
    fun i86() = _ADD(Z) //ADD A, (HL)
    fun i87() = _ADD(A) //ADD A, A

    fun iC6() = _ADD(readPC()) //ADD, n

    fun i09() = _ADDHL(BC) //ADD HL, BC
    fun i19() = _ADDHL(DE) //ADD HL, DE
    fun i29() = _ADDHL(HL) //ADD HL, HL
    fun i39() = _ADDHL(SP) //ADD HL, SP

    fun i88() = _ADC(B) //ADC A, B
    fun i89() = _ADC(C) //ADC A, C
    fun i8A() = _ADC(D) //ADC A, D
    fun i8B() = _ADC(E) //ADC A, E
    fun i8C() = _ADC(H) //ADC A, H
    fun i8D() = _ADC(L) //ADC A, L
    fun i8E() = _ADC(Z) //ADC A, (HL)
    fun i8F() = _ADC(A) //ADC A, A
    fun iCE() = _ADC(readPC()) //ADC A, n

    fun iD6() = _SUB(readPC()) //SUB A, n
    fun i90() = _SUB(B) //SUB A, B
    fun i91() = _SUB(C) //SUB A, C
    fun i92() = _SUB(D) //SUB A, D
    fun i93() = _SUB(E) //SUB A, E
    fun i94() = _SUB(H) //SUB A, H
    fun i95() = _SUB(L) //SUB A, L
    fun i96() = _SUB(Z) //SUB A, _HL_
    fun i97() = _SUB(A) //SUB A, A

    fun i98() = _SBC(B) //SBC A, B
    fun i99() = _SBC(C) //SBC A, C
    fun i9A() = _SBC(D) //SBC A, D
    fun i9B() = _SBC(E) //SBC A, E
    fun i9C() = _SBC(H) //SBC A, H
    fun i9D() = _SBC(L) //SBC A, L
    fun i9E() = _SBC(Z) //SBC A, (HL)
    fun i9F() = _SBC(A) //SBC A, A
    fun iDE() = _SBC(readPC()) //SBC A, n

    fun iA0() = _AND(B) //AND B
    fun iA1() = _AND(C) //AND C
    fun iA2() = _AND(D) //AND D
    fun iA3() = _AND(E) //AND E
    fun iA4() = _AND(H) //AND H
    fun iA5() = _AND(L) //AND L
    fun iA6() = _AND(Z) //AND (HL)
    fun iA7() = _AND(A) //AND A
    fun iE6() = _AND(readPC()) //AND n

    fun iA8() = _XOR(B) //XOR B
    fun iA9() = _XOR(C) //XOR C
    fun iAA() = _XOR(D) //XOR D
    fun iAB() = _XOR(E) //XOR E
    fun iAC() = _XOR(H) //XOR H
    fun iAD() = _XOR(L) //XOR L
    fun iAE() = _XOR(Z) //XOR (HL)
    fun iAF() = _XOR(A) //XOR A
    fun iEE() = _XOR(readPC()) //XOR n

    fun iB0() = _OR(B) //OR B
    fun iB1() = _OR(C) //OR C
    fun iB2() = _OR(D) //OR D
    fun iB3() = _OR(E) //OR E
    fun iB4() = _OR(H) //OR H
    fun iB5() = _OR(L) //OR L
    fun iB6() = _OR(Z) //OR (HL)
    fun iB7() = _OR(A) //OR A
    fun iF6() = _OR(readPC()) //OR n

    fun iB8() = _CP(B) //CP B
    fun iB9() = _CP(C) //CP C
    fun iBA() = _CP(D) //CP D
    fun iBB() = _CP(E) //CP E
    fun iBC() = _CP(H) //CP H
    fun iBD() = _CP(L) //CP L
    fun iBE() = _CP(Z) //CP (HL)
    fun iBF() = _CP(A) //CP A
    fun iFE() = _CP(readPC()) //CP n

    fun iCB() {
        val opcode = _PC_
        PC = (PC + 1) and 0xFFFF
        CPUTicks += TickTableCB[opcode]
        when (opcode) {
            0x00 -> iCB00();0x01 -> iCB01();0x02 -> iCB02();0x03 -> iCB03();0x04 -> iCB04();0x05 -> iCB05();0x06 -> iCB06()
            0x07 -> iCB07();0x08 -> iCB08();0x09 -> iCB09();0x0A -> iCB0A();0x0B -> iCB0B();0x0C -> iCB0C();0x0D -> iCB0D()
            0x0E -> iCB0E();0x0F -> iCB0F();0x10 -> iCB10();0x11 -> iCB11();0x12 -> iCB12();0x13 -> iCB13();0x14 -> iCB14()
            0x15 -> iCB15();0x16 -> iCB16();0x17 -> iCB17();0x18 -> iCB18();0x19 -> iCB19();0x1A -> iCB1A();0x1B -> iCB1B()
            0x1C -> iCB1C();0x1D -> iCB1D();0x1E -> iCB1E();0x1F -> iCB1F();0x20 -> iCB20();0x21 -> iCB21();0x22 -> iCB22()
            0x23 -> iCB23();0x24 -> iCB24();0x25 -> iCB25();0x26 -> iCB26();0x27 -> iCB27();0x28 -> iCB28();0x29 -> iCB29()
            0x2A -> iCB2A();0x2B -> iCB2B();0x2C -> iCB2C();0x2D -> iCB2D();0x2E -> iCB2E();0x2F -> iCB2F();0x30 -> iCB30()
            0x31 -> iCB31();0x32 -> iCB32();0x33 -> iCB33();0x34 -> iCB34();0x35 -> iCB35();0x36 -> iCB36();0x37 -> iCB37()
            0x38 -> iCB38();0x39 -> iCB39();0x3A -> iCB3A();0x3B -> iCB3B();0x3C -> iCB3C();0x3D -> iCB3D();0x3E -> iCB3E()
            0x3F -> iCB3F();0x40 -> iCB40();0x41 -> iCB41();0x42 -> iCB42();0x43 -> iCB43();0x44 -> iCB44();0x45 -> iCB45()
            0x46 -> iCB46();0x47 -> iCB47();0x48 -> iCB48();0x49 -> iCB49();0x4A -> iCB4A();0x4B -> iCB4B();0x4C -> iCB4C()
            0x4D -> iCB4D();0x4E -> iCB4E();0x4F -> iCB4F();0x50 -> iCB50();0x51 -> iCB51();0x52 -> iCB52();0x53 -> iCB53()
            0x54 -> iCB54();0x55 -> iCB55();0x56 -> iCB56();0x57 -> iCB57();0x58 -> iCB58();0x59 -> iCB59();0x5A -> iCB5A()
            0x5B -> iCB5B();0x5C -> iCB5C();0x5D -> iCB5D();0x5E -> iCB5E();0x5F -> iCB5F();0x60 -> iCB60();0x61 -> iCB61()
            0x62 -> iCB62();0x63 -> iCB63();0x64 -> iCB64();0x65 -> iCB65();0x66 -> iCB66();0x67 -> iCB67();0x68 -> iCB68()
            0x69 -> iCB69();0x6A -> iCB6A();0x6B -> iCB6B();0x6C -> iCB6C();0x6D -> iCB6D();0x6E -> iCB6E();0x6F -> iCB6F()
            0x70 -> iCB70();0x71 -> iCB71();0x72 -> iCB72();0x73 -> iCB73();0x74 -> iCB74();0x75 -> iCB75();0x76 -> iCB76()
            0x77 -> iCB77();0x78 -> iCB78();0x79 -> iCB79();0x7A -> iCB7A();0x7B -> iCB7B();0x7C -> iCB7C();0x7D -> iCB7D()
            0x7E -> iCB7E();0x7F -> iCB7F();0x80 -> iCB80();0x81 -> iCB81();0x82 -> iCB82();0x83 -> iCB83();0x84 -> iCB84()
            0x85 -> iCB85();0x86 -> iCB86();0x87 -> iCB87();0x88 -> iCB88();0x89 -> iCB89();0x8A -> iCB8A();0x8B -> iCB8B()
            0x8C -> iCB8C();0x8D -> iCB8D();0x8E -> iCB8E();0x8F -> iCB8F();0x90 -> iCB90();0x91 -> iCB91();0x92 -> iCB92()
            0x93 -> iCB93();0x94 -> iCB94();0x95 -> iCB95();0x96 -> iCB96();0x97 -> iCB97();0x98 -> iCB98();0x99 -> iCB99()
            0x9A -> iCB9A();0x9B -> iCB9B();0x9C -> iCB9C();0x9D -> iCB9D();0x9E -> iCB9E();0x9F -> iCB9F();0xA0 -> iCBA0()
            0xA1 -> iCBA1();0xA2 -> iCBA2();0xA3 -> iCBA3();0xA4 -> iCBA4();0xA5 -> iCBA5();0xA6 -> iCBA6();0xA7 -> iCBA7()
            0xA8 -> iCBA8();0xA9 -> iCBA9();0xAA -> iCBAA();0xAB -> iCBAB();0xAC -> iCBAC();0xAD -> iCBAD();0xAE -> iCBAE()
            0xAF -> iCBAF();0xB0 -> iCBB0();0xB1 -> iCBB1();0xB2 -> iCBB2();0xB3 -> iCBB3();0xB4 -> iCBB4();0xB5 -> iCBB5()
            0xB6 -> iCBB6();0xB7 -> iCBB7();0xB8 -> iCBB8();0xB9 -> iCBB9();0xBA -> iCBBA();0xBB -> iCBBB();0xBC -> iCBBC()
            0xBD -> iCBBD();0xBE -> iCBBE();0xBF -> iCBBF();0xC0 -> iCBC0();0xC1 -> iCBC1();0xC2 -> iCBC2();0xC3 -> iCBC3()
            0xC4 -> iCBC4();0xC5 -> iCBC5();0xC6 -> iCBC6();0xC7 -> iCBC7();0xC8 -> iCBC8();0xC9 -> iCBC9();0xCA -> iCBCA()
            0xCB -> iCBCB();0xCC -> iCBCC();0xCD -> iCBCD();0xCE -> iCBCE();0xCF -> iCBCF();0xD0 -> iCBD0();0xD1 -> iCBD1()
            0xD2 -> iCBD2();0xD3 -> iCBD3();0xD4 -> iCBD4();0xD5 -> iCBD5();0xD6 -> iCBD6();0xD7 -> iCBD7();0xD8 -> iCBD8()
            0xD9 -> iCBD9();0xDA -> iCBDA();0xDB -> iCBDB();0xDC -> iCBDC();0xDD -> iCBDD();0xDE -> iCBDE();0xDF -> iCBDF()
            0xE0 -> iCBE0();0xE1 -> iCBE1();0xE2 -> iCBE2();0xE3 -> iCBE3();0xE4 -> iCBE4();0xE5 -> iCBE5();0xE6 -> iCBE6()
            0xE7 -> iCBE7();0xE8 -> iCBE8();0xE9 -> iCBE9();0xEA -> iCBEA();0xEB -> iCBEB();0xEC -> iCBEC();0xED -> iCBED()
            0xEE -> iCBEE();0xEF -> iCBEF();0xF0 -> iCBF0();0xF1 -> iCBF1();0xF2 -> iCBF2();0xF3 -> iCBF3();0xF4 -> iCBF4()
            0xF5 -> iCBF5();0xF6 -> iCBF6();0xF7 -> iCBF7();0xF8 -> iCBF8();0xF9 -> iCBF9();0xFA -> iCBFA();0xFB -> iCBFB()
            0xFC -> iCBFC();0xFD -> iCBFD();0xFE -> iCBFE();0xFF -> iCBFF()
        }
    }

    fun iCC() = _CALL_IF(FZ) //CALL FZ, nn
    fun iCD() = _CALL_IF(true) //CALL nn
    fun iC4() = _CALL_IF(!FZ) //CALL !FZ, nn
    fun iDC() = _CALL_IF(FC) //CALL FC, nn
    fun iD4() = _CALL_IF(!FC) //CALL !FC, nn

    fun iC9() = _RET_IF(true) //RET
    fun iC0() = _RET_IF(!FZ) //RET !FZ
    fun iC8() = _RET_IF(FZ) //RET FZ
    fun iD0() = _RET_IF(!FC) //RET !FC
    fun iD8() = _RET_IF(FC) //RET FC

    fun iC2() = _JP_IF(!FZ) //JP !FZ, nn
    fun iC3() = _JP_IF(true) //JP nn
    fun iCA() = _JP_IF(FZ) //JP FZ, nn
    fun iD2() = _JP_IF(!FC) //JP !FC, nn
    fun iDA() = _JP_IF(FC) //JP FC, nn
    fun iE9() = run { PC = HL } //JP, (HL)

    fun i18() = _JR_IF(true) //JR n // @TODO: Fix CPUTicks
    fun i20() = _JR_IF(!FZ) //JR NZ, n
    fun i28() = _JR_IF(FZ) //JR Z, n
    fun i30() = _JR_IF(!FC) //JR NC, n
    fun i38() = _JR_IF(FC) //JR C, n

    fun iD9() = run { PC = readSP2(); IRQEnableDelay = if (IRQEnableDelay == 2 || _PC_ == 0x76) 1 else 2 } //RETI
    fun iE0() = run { memoryHighWrite(readPC(), A); } //LDH (n), A
    fun iE2() = run { memoryHighWrite(C, A) } //LD (0xFF00 + C), A

    fun iC5() = run { writeSP2(AF) } //PUSH BC
    fun iD5() = run { writeSP2(DE) } //PUSH DE
    fun iE5() = run { writeSP2(HL) } //PUSH HL
    fun iF5() = run { writeSP2(AF) } //PUSH AF

    fun iC1() = run { BC = readSP2() } //POP BC
    fun iD1() = run { DE = readSP2() } //POP DE
    fun iE1() = run { HL = readSP2() } //POP HL
    fun iF1() = run { AF = readSP2() } //POP AF

    fun iC7() = _RST(0x00) //RST 0x0
    fun iCF() = _RST(0x08) //RST 0x08
    fun iDF() = _RST(0x18) //RST 0x18
    fun iD7() = _RST(0x10) //RST 0x10
    fun iE7() = _RST(0x20) //RST 0x20
    fun iEF() = _RST(0x28) //RST 0x28
    fun iF7() = _RST(0x30) //RST 0x30
    fun iFF() = _RST(0x38) //RST 0x38

    //ADD SP, n
    fun iE8() {
        val t1 = sreadPC()
        val tv = (SP + t1) and 0xFFFF
        val t2 = SP xor t1 xor tv
        SP = tv
        FC = ((t2 and 0x100) == 0x100)
        FH = ((t2 and 0x10) == 0x10)
        FZ = false
        FS = false
    }

    fun iEA() = run { memoryWrite(readPC(), A) } //LD n, A
    fun iF0() = run { A = memoryReadHigh(readPC()) } //LDH A, (n)
    fun iF2() = run { A = memoryReadHigh(C) } //LD A, (0xFF00 + C)
    fun iF3() = run { IME = false; IRQEnableDelay = 0 } //DI
    fun iF8() = run {
        val t1 = sreadPC(); HL = (SP + t1) and 0xFFFF;
        val tv = SP xor t1 xor HL; FC = (tv and 0x100) != 0; FH = (tv and 0x10) != 0; FZ = false; FS = false
    } //LDHL SP, n

    fun iF9() = run { SP = HL } //LD SP, HL
    fun iFA() = run { A = memoryRead(_PC2_); PC = (PC + 2) and 0xFFFF } //LD A, (nn)
    fun iFB() = run { IRQEnableDelay = if (IRQEnableDelay == 2 || _PC_ == 0x76) 1 else 2 } //EI //Immediate for HALT:

    fun iCB00() = run { B = _RLC(B) } //RLC B
    fun iCB01() = run { C = _RLC(C) } //RLC C
    fun iCB02() = run { D = _RLC(D) } //RLC D
    fun iCB03() = run { E = _RLC(E) } //RLC E
    fun iCB04() = run { H = _RLC(H) } //RLC H
    fun iCB05() = run { L = _RLC(L) } //RLC L
    fun iCB06() = run { Z = _RLC(Z) } //RLC (HL)
    fun iCB07() = run { A = _RLC(A) } //RLC A

    fun iCB08() = run { B = _RRC(B) } //RRC B
    fun iCB09() = run { C = _RRC(C) } //RRC C
    fun iCB0A() = run { D = _RRC(D) } //RRC D
    fun iCB0B() = run { E = _RRC(E) } //RRC E
    fun iCB0C() = run { H = _RRC(H) } //RRC H
    fun iCB0D() = run { L = _RRC(L) } //RRC L
    fun iCB0E() = run { Z = _RRC(Z) } //RRC (HL)
    fun iCB0F() = run { A = _RRC(A) } //RRC A

    fun iCB10() = run { B = _RL(B) } //RL B
    fun iCB11() = run { C = _RL(C) } //RL C
    fun iCB12() = run { D = _RL(D) } //RL D
    fun iCB13() = run { E = _RL(E) } //RL E
    fun iCB14() = run { H = _RL(H) } //RL H
    fun iCB15() = run { L = _RL(L) } //RL L
    fun iCB16() = run { Z = _RL(Z) } //RL (HL)
    fun iCB17() = run { A = _RL(A) } //RL A

    fun iCB18() = run { B = _RR(B) } //RR B
    fun iCB19() = run { C = _RR(C) } //RR C
    fun iCB1A() = run { D = _RR(D) } //RR D
    fun iCB1B() = run { E = _RR(E) } //RR E
    fun iCB1C() = run { H = _RR(H) } //RR H
    fun iCB1D() = run { L = _RR(L) } //RR L
    fun iCB1E() = run { Z = _RR(Z) } //RR (HL)
    fun iCB1F() = run { A = _RR(A) } //RR A

    fun iCB20() = run { B = _SLA(B) } //SLA B
    fun iCB21() = run { C = _SLA(C) } //SLA C
    fun iCB22() = run { D = _SLA(D) } //SLA D
    fun iCB23() = run { E = _SLA(E) } //SLA E
    fun iCB24() = run { H = _SLA(H) } //SLA H
    fun iCB25() = run { L = _SLA(L) } //SLA L
    fun iCB26() = run { Z = _SLA(Z) } //SLA (HL)
    fun iCB27() = run { A = _SLA(A) } //SLA A

    fun iCB28() = run { B = _SRA(B) } //SRA B
    fun iCB29() = run { C = _SRA(C) } //SRA C
    fun iCB2A() = run { D = _SRA(D) } //SRA D
    fun iCB2B() = run { E = _SRA(E) } //SRA E
    fun iCB2C() = run { H = _SRA(H) } //SRA H
    fun iCB2D() = run { L = _SRA(L) } //SRA L
    fun iCB2E() = run { Z = _SRA(Z) } //SRA (HL)
    fun iCB2F() = run { A = _SRA(A) } //SRA A

    fun iCB30() = run { B = _SWAP(B) } //SWAP B
    fun iCB31() = run { C = _SWAP(C) } //SWAP C
    fun iCB32() = run { D = _SWAP(D) } //SWAP D
    fun iCB33() = run { E = _SWAP(E) } //SWAP E
    fun iCB34() = run { H = _SWAP(H) } //SWAP H
    fun iCB35() = run { L = _SWAP(L) } //SWAP L
    fun iCB36() = run { Z = _SWAP(Z) } //SWAP (HL)
    fun iCB37() = run { A = _SWAP(A) } //SWAP A

    fun iCB38() = run { B = _SRL(B) } //SRL B
    fun iCB39() = run { C = _SRL(C) } //SRL C
    fun iCB3A() = run { D = _SRL(D) } //SRL D
    fun iCB3B() = run { E = _SRL(E) } //SRL E
    fun iCB3C() = run { H = _SRL(H) } //SRL H
    fun iCB3D() = run { L = _SRL(L) } //SRL L
    fun iCB3E() = run { Z = _SRL(Z) } //SRL (HL)
    fun iCB3F() = run { A = _SRL(A) } //SRL A

    fun iCB40() = _BIT(0, B) //BIT 0, B
    fun iCB41() = _BIT(0, C) //BIT 0, C
    fun iCB42() = _BIT(0, D) //BIT 0, D
    fun iCB43() = _BIT(0, E) //BIT 0, E
    fun iCB44() = _BIT(0, H) //BIT 0, H
    fun iCB45() = _BIT(0, L) //BIT 0, L
    fun iCB46() = _BIT(0, Z) //BIT 0, (HL)
    fun iCB47() = _BIT(0, A) //BIT 0, A

    fun iCB48() = _BIT(1, B) //BIT 1, B
    fun iCB49() = _BIT(1, C) //BIT 1, C
    fun iCB4A() = _BIT(1, D) //BIT 1, D
    fun iCB4B() = _BIT(1, E) //BIT 1, E
    fun iCB4C() = _BIT(1, H) //BIT 1, H
    fun iCB4D() = _BIT(1, L) //BIT 1, L
    fun iCB4E() = _BIT(1, Z) //BIT 1, (HL)
    fun iCB4F() = _BIT(1, A) //BIT 1, A

    fun iCB50() = _BIT(2, B) //BIT 2, B
    fun iCB51() = _BIT(2, C) //BIT 2, C
    fun iCB52() = _BIT(2, D) //BIT 2, D
    fun iCB53() = _BIT(2, E) //BIT 2, E
    fun iCB54() = _BIT(2, H) //BIT 2, H
    fun iCB55() = _BIT(2, L) //BIT 2, L
    fun iCB56() = _BIT(2, Z) //BIT 2, (HL)
    fun iCB57() = _BIT(2, A) //BIT 2, A

    fun iCB58() = _BIT(3, B) //BIT 3, B
    fun iCB59() = _BIT(3, C) //BIT 3, C
    fun iCB5A() = _BIT(3, D) //BIT 3, D
    fun iCB5B() = _BIT(3, E) //BIT 3, E
    fun iCB5C() = _BIT(3, H) //BIT 3, H
    fun iCB5D() = _BIT(3, L) //BIT 3, L
    fun iCB5E() = _BIT(3, Z) //BIT 3, (HL)
    fun iCB5F() = _BIT(3, A) //BIT 3, A

    fun iCB60() = _BIT(4, B) //BIT 4, B
    fun iCB61() = _BIT(4, C) //BIT 4, C
    fun iCB62() = _BIT(4, D) //BIT 4, D
    fun iCB63() = _BIT(4, E) //BIT 4, E
    fun iCB64() = _BIT(4, H) //BIT 4, H
    fun iCB65() = _BIT(4, L) //BIT 4, L
    fun iCB66() = _BIT(4, Z) //BIT 4, (HL)
    fun iCB67() = _BIT(4, A) //BIT 4, A

    fun iCB68() = _BIT(5, B) //BIT 5, B
    fun iCB69() = _BIT(5, C) //BIT 5, C
    fun iCB6A() = _BIT(5, D) //BIT 5, D
    fun iCB6B() = _BIT(5, E) //BIT 5, E
    fun iCB6C() = _BIT(5, H) //BIT 5, H
    fun iCB6D() = _BIT(5, L) //BIT 5, L
    fun iCB6E() = _BIT(5, Z) //BIT 5, (HL)
    fun iCB6F() = _BIT(5, A) //BIT 5, A

    fun iCB70() = _BIT(6, B) //BIT 6, B
    fun iCB71() = _BIT(6, C) //BIT 6, C
    fun iCB72() = _BIT(6, D) //BIT 6, D
    fun iCB73() = _BIT(6, E) //BIT 6, E
    fun iCB74() = _BIT(6, H) //BIT 6, H
    fun iCB75() = _BIT(6, L) //BIT 6, L
    fun iCB76() = _BIT(6, Z) //BIT 6, (HL)
    fun iCB77() = _BIT(6, A) //BIT 6, A

    fun iCB78() = _BIT(7, B) //BIT 7, B
    fun iCB79() = _BIT(7, C) //BIT 7, C
    fun iCB7A() = _BIT(7, D) //BIT 7, D
    fun iCB7B() = _BIT(7, E) //BIT 7, E
    fun iCB7C() = _BIT(7, H) //BIT 7, H
    fun iCB7D() = _BIT(7, L) //BIT 7, L
    fun iCB7E() = _BIT(7, Z) //BIT 7, (HL)
    fun iCB7F() = _BIT(7, A) //BIT 7, A

    fun iCB80() = run { B = _RES(0, B) } //RES 0, B
    fun iCB81() = run { C = _RES(0, C) } //RES 0, C
    fun iCB82() = run { D = _RES(0, D) } //RES 0, D
    fun iCB83() = run { E = _RES(0, E) } //RES 0, E
    fun iCB84() = run { H = _RES(0, H) } //RES 0, H
    fun iCB85() = run { L = _RES(0, L) } //RES 0, L
    fun iCB86() = run { Z = _RES(0, Z) } //RES 0, (HL)
    fun iCB87() = run { A = _RES(0, A) } //RES 0, A

    fun iCB88() = run { B = _RES(1, B) } //RES 1, B
    fun iCB89() = run { C = _RES(1, C) } //RES 1, C
    fun iCB8A() = run { D = _RES(1, D) } //RES 1, D
    fun iCB8B() = run { E = _RES(1, E) } //RES 1, E
    fun iCB8C() = run { H = _RES(1, H) } //RES 1, H
    fun iCB8D() = run { L = _RES(1, L) } //RES 1, L
    fun iCB8E() = run { Z = _RES(1, Z) } //RES 1, (HL)
    fun iCB8F() = run { A = _RES(1, A) } //RES 1, A

    fun iCB90() = run { B = _RES(2, B) } //RES 2, B
    fun iCB91() = run { C = _RES(2, C) } //RES 2, C
    fun iCB92() = run { D = _RES(2, D) } //RES 2, D
    fun iCB93() = run { E = _RES(2, E) } //RES 2, E
    fun iCB94() = run { H = _RES(2, H) } //RES 2, H
    fun iCB95() = run { L = _RES(2, L) } //RES 2, L
    fun iCB96() = run { Z = _RES(2, Z) } //RES 2, (HL)
    fun iCB97() = run { A = _RES(2, A) } //RES 2, A

    fun iCB98() = run { B = _RES(3, B) } //RES 3, B
    fun iCB99() = run { C = _RES(3, C) } //RES 3, C
    fun iCB9A() = run { D = _RES(3, D) } //RES 3, D
    fun iCB9B() = run { E = _RES(3, E) } //RES 3, E
    fun iCB9C() = run { H = _RES(3, H) } //RES 3, H
    fun iCB9D() = run { L = _RES(3, L) } //RES 3, L
    fun iCB9E() = run { Z = _RES(3, Z) } //RES 3, (HL)
    fun iCB9F() = run { A = _RES(3, A) } //RES 3, A

    fun iCBA0() = run { B = _RES(4, B) } //RES 4, B
    fun iCBA1() = run { C = _RES(4, C) } //RES 4, C
    fun iCBA2() = run { D = _RES(4, D) } //RES 4, D
    fun iCBA3() = run { E = _RES(4, E) } //RES 4, E
    fun iCBA4() = run { H = _RES(4, H) } //RES 4, H
    fun iCBA5() = run { L = _RES(4, L) } //RES 4, L
    fun iCBA6() = run { Z = _RES(4, Z) } //RES 4, (HL)
    fun iCBA7() = run { A = _RES(4, A) } //RES 4, A

    fun iCBA8() = run { B = _RES(5, B) } //RES 5, B
    fun iCBA9() = run { C = _RES(5, C) } //RES 5, C
    fun iCBAA() = run { D = _RES(5, D) } //RES 5, D
    fun iCBAB() = run { E = _RES(5, E) } //RES 5, E
    fun iCBAC() = run { H = _RES(5, H) } //RES 5, H
    fun iCBAD() = run { L = _RES(5, L) } //RES 5, L
    fun iCBAE() = run { Z = _RES(5, Z) } //RES 5, (HL)
    fun iCBAF() = run { A = _RES(5, A) } //RES 5, A

    fun iCBB0() = run { B = _RES(6, B) } //RES 6, B
    fun iCBB1() = run { C = _RES(6, C) } //RES 6, C
    fun iCBB2() = run { D = _RES(6, D) } //RES 6, D
    fun iCBB3() = run { E = _RES(6, E) } //RES 6, E
    fun iCBB4() = run { H = _RES(6, H) } //RES 6, H
    fun iCBB5() = run { L = _RES(6, L) } //RES 6, L
    fun iCBB6() = run { Z = _RES(6, Z) } //RES 6, (HL)
    fun iCBB7() = run { A = _RES(6, A) } //RES 6, A

    fun iCBB8() = run { B = _RES(7, B) } //RES 7, B
    fun iCBB9() = run { C = _RES(7, C) } //RES 7, C
    fun iCBBA() = run { D = _RES(7, D) } //RES 7, D
    fun iCBBB() = run { E = _RES(7, E) } //RES 7, E
    fun iCBBC() = run { H = _RES(7, H) } //RES 7, H
    fun iCBBD() = run { L = _RES(7, L) } //RES 7, L
    fun iCBBE() = run { Z = _RES(7, Z) } //RES 7, (HL)
    fun iCBBF() = run { A = _RES(7, A) } //RES 7, A

    fun iCBC0() = run { B = _SET(0, B) } //SET 0, B
    fun iCBC1() = run { C = _SET(0, C) } //SET 0, C
    fun iCBC2() = run { D = _SET(0, D) } //SET 0, D
    fun iCBC3() = run { E = _SET(0, E) } //SET 0, E
    fun iCBC4() = run { H = _SET(0, H) } //SET 0, H
    fun iCBC5() = run { L = _SET(0, L) } //SET 0, L
    fun iCBC6() = run { Z = _SET(0, Z) } //SET 0, (HL)
    fun iCBC7() = run { A = _SET(0, A) } //SET 0, A

    fun iCBC8() = run { B = _SET(1, B) } //SET 1, B
    fun iCBC9() = run { C = _SET(1, C) } //SET 1, C
    fun iCBCA() = run { D = _SET(1, D) } //SET 1, D
    fun iCBCB() = run { E = _SET(1, E) } //SET 1, E
    fun iCBCC() = run { H = _SET(1, H) } //SET 1, H
    fun iCBCD() = run { L = _SET(1, L) } //SET 1, L
    fun iCBCE() = run { Z = _SET(1, Z) } //SET 1, (HL)
    fun iCBCF() = run { A = _SET(1, A) } //SET 1, A

    fun iCBD0() = run { B = _SET(2, B) } //SET 2, B
    fun iCBD1() = run { C = _SET(2, C) } //SET 2, C
    fun iCBD2() = run { D = _SET(2, D) } //SET 2, D
    fun iCBD3() = run { E = _SET(2, E) } //SET 2, E
    fun iCBD4() = run { H = _SET(2, H) } //SET 2, H
    fun iCBD5() = run { L = _SET(2, L) } //SET 2, L
    fun iCBD6() = run { Z = _SET(2, Z) } //SET 2, (HL)
    fun iCBD7() = run { A = _SET(2, A) } //SET 2, A

    fun iCBD8() = run { B = _SET(3, B) } //SET 3, B
    fun iCBD9() = run { C = _SET(3, C) } //SET 3, C
    fun iCBDA() = run { D = _SET(3, D) } //SET 3, D
    fun iCBDB() = run { E = _SET(3, E) } //SET 3, E
    fun iCBDC() = run { H = _SET(3, H) } //SET 3, H
    fun iCBDD() = run { L = _SET(3, L) } //SET 3, L
    fun iCBDE() = run { Z = _SET(3, Z) } //SET 3, (HL)
    fun iCBDF() = run { A = _SET(3, A) } //SET 3, A

    fun iCBE0() = run { B = _SET(4, B) } //SET 4, B
    fun iCBE1() = run { C = _SET(4, C) } //SET 4, C
    fun iCBE2() = run { D = _SET(4, D) } //SET 4, D
    fun iCBE3() = run { E = _SET(4, E) } //SET 4, E
    fun iCBE4() = run { H = _SET(4, H) } //SET 4, H
    fun iCBE5() = run { L = _SET(4, L) } //SET 4, L
    fun iCBE6() = run { Z = _SET(4, Z) } //SET 4, (HL)
    fun iCBE7() = run { A = _SET(4, A) } //SET 4, A

    fun iCBE8() = run { B = _SET(5, B) } //SET 5, B
    fun iCBE9() = run { C = _SET(5, C) } //SET 5, C
    fun iCBEA() = run { D = _SET(5, D) } //SET 5, D
    fun iCBEB() = run { E = _SET(5, E) } //SET 5, E
    fun iCBEC() = run { H = _SET(5, H) } //SET 5, H
    fun iCBED() = run { L = _SET(5, L) } //SET 5, L
    fun iCBEE() = run { Z = _SET(5, Z) } //SET 5, (HL)
    fun iCBEF() = run { A = _SET(5, A) } //SET 5, A

    fun iCBF0() = run { B = _SET(6, B) } //SET 6, B
    fun iCBF1() = run { C = _SET(6, C) } //SET 6, C
    fun iCBF2() = run { D = _SET(6, D) } //SET 6, D
    fun iCBF3() = run { E = _SET(6, E) } //SET 6, E
    fun iCBF4() = run { H = _SET(6, H) } //SET 6, H
    fun iCBF5() = run { L = _SET(6, L) } //SET 6, L
    fun iCBF6() = run { Z = _SET(6, Z) } //SET 6, (HL)
    fun iCBF7() = run { A = _SET(6, A) } //SET 6, A

    fun iCBF8() = run { B = _SET(7, B) } //SET 7, B
    fun iCBF9() = run { C = _SET(7, C) } //SET 7, C
    fun iCBFA() = run { D = _SET(7, D) } //SET 7, D
    fun iCBFB() = run { E = _SET(7, E) } //SET 7, E
    fun iCBFC() = run { H = _SET(7, H) } //SET 7, H
    fun iCBFD() = run { L = _SET(7, L) } //SET 7, L
    fun iCBFE() = run { Z = _SET(7, Z) } //SET 7, (HL)
    fun iCBFF() = run { A = _SET(7, A) } //SET 7, A

    fun iFC() = illegal(this, 0xFC) //0xFC - Illegal
    fun iFD() = illegal(this, 0xFD) //0xFD - Illegal
    fun iF4() = illegal(this, 0xF4) //0xF4 - Illegal
    fun iEB() = illegal(this, 0xEB) //0xEB - Illegal
    fun iEC() = illegal(this, 0xEC) //0xEC - Illegal
    fun iED() = illegal(this, 0xED) //0xED - Illegal
    fun iE3() = illegal(this, 0xE3) //0xE3 - Illegal
    fun iE4() = illegal(this, 0xE4) //0xE4 - Illegal
    fun iD3() = illegal(this, 0xD3) //0xD3 - Illegal
    fun iDB() = illegal(this, 0xDB) //0xDB - Illegal
    fun iDD() = illegal(this, 0xDD) //0xDD - Illegal
}

private val TickTableNone = intArrayOf(
    4, 12, 8, 8, 4, 4, 8, 4, 20, 8, 8, 8, 4, 4, 8, 4,
    4, 12, 8, 8, 4, 4, 8, 4, 12, 8, 8, 8, 4, 4, 8, 4,
    8, 12, 8, 8, 4, 4, 8, 4, 8, 8, 8, 8, 4, 4, 8, 4,
    8, 12, 8, 8, 12, 12, 12, 4, 8, 8, 8, 8, 4, 4, 8, 4,
    4, 4, 4, 4, 4, 4, 8, 4, 4, 4, 4, 4, 4, 4, 8, 4,
    4, 4, 4, 4, 4, 4, 8, 4, 4, 4, 4, 4, 4, 4, 8, 4,
    4, 4, 4, 4, 4, 4, 8, 4, 4, 4, 4, 4, 4, 4, 8, 4,
    8, 8, 8, 8, 8, 8, 4, 8, 4, 4, 4, 4, 4, 4, 8, 4,
    4, 4, 4, 4, 4, 4, 8, 4, 4, 4, 4, 4, 4, 4, 8, 4,
    4, 4, 4, 4, 4, 4, 8, 4, 4, 4, 4, 4, 4, 4, 8, 4,
    4, 4, 4, 4, 4, 4, 8, 4, 4, 4, 4, 4, 4, 4, 8, 4,
    4, 4, 4, 4, 4, 4, 8, 4, 4, 4, 4, 4, 4, 4, 8, 4,
    8, 12, 12, 16, 12, 16, 8, 16, 8, 16, 12, 0, 12, 24, 8, 16,
    8, 12, 12, 4, 12, 16, 8, 16, 8, 16, 12, 4, 12, 4, 8, 16,
    12, 12, 8, 4, 4, 16, 8, 16, 16, 4, 16, 4, 4, 4, 8, 16,
    12, 12, 8, 4, 4, 16, 8, 16, 12, 8, 16, 4, 0, 4, 8, 16
)

private val TickTableCB = intArrayOf(
    8, 8, 8, 8, 8, 8, 16, 8, 8, 8, 8, 8, 8, 8, 16, 8,
    8, 8, 8, 8, 8, 8, 16, 8, 8, 8, 8, 8, 8, 8, 16, 8,
    8, 8, 8, 8, 8, 8, 16, 8, 8, 8, 8, 8, 8, 8, 16, 8,
    8, 8, 8, 8, 8, 8, 16, 8, 8, 8, 8, 8, 8, 8, 16, 8,
    8, 8, 8, 8, 8, 8, 12, 8, 8, 8, 8, 8, 8, 8, 12, 8,
    8, 8, 8, 8, 8, 8, 12, 8, 8, 8, 8, 8, 8, 8, 12, 8,
    8, 8, 8, 8, 8, 8, 12, 8, 8, 8, 8, 8, 8, 8, 12, 8,
    8, 8, 8, 8, 8, 8, 12, 8, 8, 8, 8, 8, 8, 8, 12, 8,
    8, 8, 8, 8, 8, 8, 16, 8, 8, 8, 8, 8, 8, 8, 16, 8,
    8, 8, 8, 8, 8, 8, 16, 8, 8, 8, 8, 8, 8, 8, 16, 8,
    8, 8, 8, 8, 8, 8, 16, 8, 8, 8, 8, 8, 8, 8, 16, 8,
    8, 8, 8, 8, 8, 8, 16, 8, 8, 8, 8, 8, 8, 8, 16, 8,
    8, 8, 8, 8, 8, 8, 16, 8, 8, 8, 8, 8, 8, 8, 16, 8,
    8, 8, 8, 8, 8, 8, 16, 8, 8, 8, 8, 8, 8, 8, 16, 8,
    8, 8, 8, 8, 8, 8, 16, 8, 8, 8, 8, 8, 8, 8, 16, 8,
    8, 8, 8, 8, 8, 8, 16, 8, 8, 8, 8, 8, 8, 8, 16, 8
)
