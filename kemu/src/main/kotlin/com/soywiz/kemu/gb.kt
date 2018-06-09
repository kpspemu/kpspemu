package com.soywiz.kemu

/*
class Gameboy : Z80Controller {
    companion object {
        const val SPEED_CHANGE = 0xFF4D
    }

    val cpu = Z80(this)
    val memory = UByteArray(0x10000)
    var currentROMBank = 0
    var numRAMBanks = 2
    var cMBC7 = false
    var cMBC3 = false

    override fun memoryRead(index: Int): Int {
        when {
            index < 0x4000 -> return memoryReadNormal(index)
            index < 0x8000 -> return this.memoryReadROM(index)
            index < 0x9800 -> return if (this.cGBC) this.VRAMDATAReadCGBCPU(index) else this.VRAMDATAReadDMGCPU(index)
            index < 0xA000 -> return if (this.cGBC) this.VRAMCHRReadCGBCPU(index) else this.VRAMCHRReadDMGCPU(index)
            index in 0xA000..49151 -> if ((this.numRAMBanks == 1 / 16 && index < 0xA200) || this.numRAMBanks >= 1) {
                when {
                    this.cMBC7 -> return this.memoryReadMBC7(index)
                    !this.cMBC3 -> return this.memoryReadMBC(index)
                    else -> return this.memoryReadMBC3(index)
                //MBC3 RTC + RAM:
                }
            } else {
                return this.memoryReadBAD(index)
            }
            index in 0xC000..57343 -> return if (!this.cGBC || index < 0xD000) this.memoryReadNormal(index) else this.memoryReadGBCMemory(index)
            index in 0xE000..65023 -> return if (!this.cGBC || index < 0xF000) this.memoryReadECHONormal(index) else this.memoryReadECHOGBCMemory(index)
            index < 0xFEA0 -> return this.memoryReadOAM(index)
            this.cGBC && index >= 0xFEA0 && index < 0xFF00 -> return this.memoryReadNormal(index)
            index >= 0xFF00 -> {

                when (index) {
                    0xFF00 -> return 0xC0 or memory[0xFF00] //JOYPAD:  //Top nibble returns as set.
                    0xFF01 -> return if (memory[0xFF02] < 0x80) memory[0xFF01] else 0xFF //SB
                    0xFF02 -> {
                        //SC
                        if (this.cGBC) {
                            this.memoryHighReader[0x02] = this.memoryReader[0xFF02] = function(parentObj, address) {
                                return ((parentObj.serialTimer <= 0) ? 0x7C : 0xFC) | parentObj.memory[0xFF02];
                            }
                        } else {
                            this.memoryHighReader[0x02] = this.memoryReader[0xFF02] = function(parentObj, address) {
                                return ((parentObj.serialTimer <= 0) ? 0x7E : 0xFE) | parentObj.memory[0xFF02];
                            }
                        }
                    }
                    0xFF03 -> return this.memoryReadBAD()
                    0xFF04 -> {
                        //DIV
                        memory[0xFF04] = (memory[0xFF04] + (parentObj.DIVTicks > > 8)) & 0xFF;
                        DIVTicks & = 0xFF;
                        return memory[0xFF04];
                    }
                    0xFF05, 0xFF06 -> return this.memoryHighReadNormal(index)
                    0xFF07 -> return 0xF8 or memory[0xFF07];
                    0xFF08, 0xFF09, 0xFF0A, 0xFF0B, 0xFF0C, 0xFF0D, 0xFF0E -> return this.memoryReadBAD(index)
                    0xFF0F -> return 0xE0 or interruptsRequested(address)
                    0xFF10 -> return 0x80 or memory[0xFF10]
                    0xFF11 -> return 0x3F or memory[0xFF11]
                    0xFF12 -> return this.memoryReadNormal(index)
                    0xFF13 -> return this.memoryReadBAD(index)
                    0xFF14 -> return 0xBF or memory[0xFF14]
                    0xFF15 -> return this.memoryReadBAD(index)
                    0xFF16 -> return 0x3F or memory[0xFF16];
                    0xFF17 -> return this.memoryReadNormal(index)
                    0xFF18 -> return this.memoryReadBAD(index)
                    0xFF19 -> return 0xBF or memory[0xFF19];
                    0xFF1A -> return 0x7F or memory[0xFF1A];
                    0xFF1B -> return this.memoryReadBAD(index)
                    0xFF1C -> return 0x9F or memory[0xFF1C];
                    0xFF1D -> return this.memoryReadBAD(index)
                    0xFF1E -> return 0xBF or memory[0xFF1E];
                    0xFF1F, 0xFF20 -> return this.memoryReadBAD(index)
                    0xFF21, 0xFF22 -> return this.memoryReadNormal(index)
                    0xFF23 -> return 0xBF or memory[0xFF23];
                    0xFF24, 0xFF25 -> return this.memoryReadNormal(index)
                    0xFF26 -> return 0x70 or memory[0xFF26];
                    0xFF27, 0xFF28, 0xFF29, 0xFF2A, 0xFF2B, 0xFF2C, 0xFF2D, 0xFF2E, 0xFF2F -> return this.memoryReadBAD(index)
                    0xFF30, 0xFF31, 0xFF32, 0xFF33, 0xFF34, 0xFF35, 0xFF36, 0xFF37, 0xFF38, 0xFF39, 0xFF3A, 0xFF3B, 0xFF3C, 0xFF3D, 0xFF3E, 0xFF3F -> if (channel3canPlay) memory[0xFF00 or (channel3lastSampleLookup ushr 1)] else memory[index];
                    0xFF40 -> return this.memoryReadNormal(index)
                    0xFF41 -> return 0x80 | parentObj.memory[0xFF41] | parentObj.modeSTAT;
                    0xFF42 -> return parentObj.backgroundY;
                    0xFF43 -> return parentObj.backgroundX;
                    0xFF44 -> return ((parentObj.LCDisOn) ? parentObj.memory[0xFF44] : 0);
                    0xFF45, 0xFF46, 0xFF47, 0xFF48, 0xFF49 -> return this.memoryReadNormal(index)
                    0xFF4A -> return parentObj.windowY;
                    0xFF4B -> return this.memoryReadNormal(index)
                    0xFF4C -> return this.memoryReadBAD(index)
                    0xFF4D -> return this.memoryReadNormal(index)
                    0xFF4E -> return this.memoryReadBAD(index)
                    0xFF4F -> return parentObj.currVRAMBank;
                    0xFF50, 0xFF51, 0xFF52, 0xFF53, 0xFF54 -> return this.memoryReadNormal(index)
                    case 0xFF55:
                    if (this.cGBC) {
                        this.memoryHighReader[0x55] = this.memoryReader[0xFF55] = function (parentObj, address) {
                            if (!parentObj.LCDisOn && parentObj.hdmaRunning) {	//Undocumented behavior alert: HDMA becomes GDMA when LCD is off (Worms Armageddon Fix).
                                //DMA
                                parentObj.DMAWrite((parentObj.memory[0xFF55] & 0x7F) + 1);
                                parentObj.memory[0xFF55] = 0xFF;	//Transfer completed.
                                parentObj.hdmaRunning = false;
                            }
                            return parentObj.memory[0xFF55];
                        }
                    }
                    else {
                        this.memoryReader[0xFF55] = this.memoryReadNormal;
                        this.memoryHighReader[0x55] = this.memoryHighReadNormal;
                    }
                    break;
                    case 0xFF56:
                    if (this.cGBC) {
                        this.memoryHighReader[0x56] = this.memoryReader[0xFF56] = function (parentObj, address) {
                            //Return IR "not connected" status:
                            return 0x3C | ((parentObj.memory[0xFF56] >= 0xC0) ? (0x2 | (parentObj.memory[0xFF56] & 0xC1)) : (parentObj.memory[0xFF56] & 0xC3));
                        }
                    }
                    else {
                        this.memoryReader[0xFF56] = this.memoryReadNormal;
                        this.memoryHighReader[0x56] = this.memoryHighReadNormal;
                    }
                    break;
                    0xFF57,0xFF58, 0xFF59, 0xFF5A, 0xFF5B, 0xFF5C, 0xFF5D, 0xFF5E, 0xFF5F, 0xFF60, 0xFF61, 0xFF62, 0xFF63, 0xFF64, 0xFF65, 0xFF66, 0xFF67 -> return this.memoryReadBAD(index)
                    case 0xFF68:
                    case 0xFF69:
                    case 0xFF6A:
                    case 0xFF6B:
                    this.memoryHighReader[index & 0xFF] = this.memoryHighReadNormal;
                    this.memoryReader[index] = this.memoryReadNormal;
                    break;
                    case 0xFF6C:
                    if (this.cGBC) {
                        this.memoryHighReader[0x6C] = this.memoryReader[0xFF6C] = function (parentObj, address) {
                            return 0xFE | parentObj.memory[0xFF6C];
                        }
                    }
                    else {
                        this.memoryHighReader[0x6C] = this.memoryReader[0xFF6C] = this.memoryReadBAD;
                    }
                    break;
                    case 0xFF6D:
                    case 0xFF6E:
                    case 0xFF6F:
                    this.memoryHighReader[index & 0xFF] = this.memoryReader[index] = this.memoryReadBAD;
                    break;
                    case 0xFF70:
                    if (this.cGBC) {
                        //SVBK
                        this.memoryHighReader[0x70] = this.memoryReader[0xFF70] = function (parentObj, address) {
                            return 0x40 | parentObj.memory[0xFF70];
                        }
                    }
                    else {
                        this.memoryHighReader[0x70] = this.memoryReader[0xFF70] = this.memoryReadBAD;
                    }
                    break;
                    case 0xFF71:
                    this.memoryHighReader[0x71] = this.memoryReader[0xFF71] = this.memoryReadBAD;
                    break;
                    case 0xFF72:
                    case 0xFF73:
                    this.memoryHighReader[index & 0xFF] = this.memoryReader[index] = this.memoryReadNormal;
                    break;
                    case 0xFF74:
                    if (this.cGBC) {
                        this.memoryHighReader[0x74] = this.memoryReader[0xFF74] = this.memoryReadNormal;
                    }
                    else {
                        this.memoryHighReader[0x74] = this.memoryReader[0xFF74] = this.memoryReadBAD;
                    }
                    break;
                    case 0xFF75:
                    this.memoryHighReader[0x75] = this.memoryReader[0xFF75] = function (parentObj, address) {
                        return 0x8F | parentObj.memory[0xFF75];
                    }
                    break;
                    case 0xFF76:
                    //Undocumented realtime PCM amplitude readback:
                    this.memoryHighReader[0x76] = this.memoryReader[0xFF76] = function (parentObj, address) {
                        parentObj.audioJIT();
                        return (parentObj.channel2envelopeVolume << 4) | parentObj.channel1envelopeVolume;
                    }
                    break;
                    case 0xFF77:
                    //Undocumented realtime PCM amplitude readback:
                    this.memoryHighReader[0x77] = this.memoryReader[0xFF77] = function (parentObj, address) {
                        parentObj.audioJIT();
                        return (parentObj.channel4envelopeVolume << 4) | parentObj.channel3envelopeVolume;
                    }
                    break;
                    case 0xFF78:
                    case 0xFF79:
                    case 0xFF7A:
                    case 0xFF7B:
                    case 0xFF7C:
                    case 0xFF7D:
                    case 0xFF7E:
                    case 0xFF7F:
                    this.memoryHighReader[index & 0xFF] = this.memoryReader[index] = this.memoryReadBAD;
                    break;
                    case 0xFFFF:
                    //IE
                    this.memoryHighReader[0xFF] = this.memoryReader[0xFFFF] = function (parentObj, address) {
                        return parentObj.interruptsEnabled;
                    }
                    break;
                    else -> return this.memoryReadNormal(index)
                }

            }
            else -> return this.memoryReadBAD(index)
        }
    }

    private fun memoryHighReadNormal(index: Int) = memory[index]
    private fun memoryReadNormal(index: Int) = memory[index]
    private fun memoryReadBAD(index: Int): Int = TODO()

    override fun memoryWrite(addr: Int, value: Int) {
        memory[addr] = value
    }

    override fun halt(cpu: Z80) {
        if ((cpu.interruptsEnabled and cpu.interruptsRequested and 0x1F) > 0) {
            if (!cGBC && !cpu.usedBootROM) {
                cpu.skipPCIncrement = true
            } else {
                cpu.CPUTicks += 4
            }
        } else {
            calculateHALTPeriod()
        }
    }

    var cGBC = false
    var doubleSpeedShifter = 0

    override fun stop(cpu: Z80) {
        if (cGBC && (memory[SPEED_CHANGE] and 0x01) == 0x01) {
            if (memory[SPEED_CHANGE] > 0x7F) {
                doubleSpeedShifter = 0
                memory[SPEED_CHANGE] = memory[SPEED_CHANGE] and 0x7F
            } else {
                doubleSpeedShifter = 1
                memory[SPEED_CHANGE] = memory[SPEED_CHANGE] or 0x80
            }
            memory[SPEED_CHANGE] = memory[SPEED_CHANGE] and 0xFE
        } else {
            TODO()
        }
    }

    fun calculateHALTPeriod(): Unit = TODO()
}

private val ffxxDump = intArrayOf(
    0x0F, 0x00, 0x7C, 0xFF, 0x00, 0x00, 0x00, 0xF8, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x01,
    0x80, 0xBF, 0xF3, 0xFF, 0xBF, 0xFF, 0x3F, 0x00, 0xFF, 0xBF, 0x7F, 0xFF, 0x9F, 0xFF, 0xBF, 0xFF,
    0xFF, 0x00, 0x00, 0xBF, 0x77, 0xF3, 0xF1, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF,
    0x91, 0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFC, 0x00, 0x00, 0x00, 0x00, 0xFF, 0x7E, 0xFF, 0xFE,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x3E, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xC0, 0xFF, 0xC1, 0x00, 0xFE, 0xFF, 0xFF, 0xFF,
    0xF8, 0xFF, 0x00, 0x00, 0x00, 0x8F, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xCE, 0xED, 0x66, 0x66, 0xCC, 0x0D, 0x00, 0x0B, 0x03, 0x73, 0x00, 0x83, 0x00, 0x0C, 0x00, 0x0D,
    0x00, 0x08, 0x11, 0x1F, 0x88, 0x89, 0x00, 0x0E, 0xDC, 0xCC, 0x6E, 0xE6, 0xDD, 0xDD, 0xD9, 0x99,
    0xBB, 0xBB, 0x67, 0x63, 0x6E, 0x0E, 0xEC, 0xCC, 0xDD, 0xDC, 0x99, 0x9F, 0xBB, 0xB9, 0x33, 0x3E,
    0x45, 0xEC, 0x52, 0xFA, 0x08, 0xB7, 0x07, 0x5D, 0x01, 0xFD, 0xC0, 0xFF, 0x08, 0xFC, 0x00, 0xE5,
    0x0B, 0xF8, 0xC2, 0xCE, 0xF4, 0xF9, 0x0F, 0x7F, 0x45, 0x6D, 0x3D, 0xFE, 0x46, 0x97, 0x33, 0x5E,
    0x08, 0xEF, 0xF1, 0xFF, 0x86, 0x83, 0x24, 0x74, 0x12, 0xFC, 0x00, 0x9F, 0xB4, 0xB7, 0x06, 0xD5,
    0xD0, 0x7A, 0x00, 0x9E, 0x04, 0x5F, 0x41, 0x2F, 0x1D, 0x77, 0x36, 0x75, 0x81, 0xAA, 0x70, 0x3A,
    0x98, 0xD1, 0x71, 0x02, 0x4D, 0x01, 0xC1, 0xFF, 0x0D, 0x00, 0xD3, 0x05, 0xF9, 0x00, 0x0B, 0x00
)
        */
