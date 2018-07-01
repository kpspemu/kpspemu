package com.soywiz.kpspemu.hle.modules


import com.soywiz.kpspemu.*
import com.soywiz.kpspemu.cpu.*
import com.soywiz.kpspemu.hle.*


class sceMpeg(emulator: Emulator) : SceModule(emulator, "sceMpeg", 0x00010011, "mpeg_vsh.prx", "sceMpegVsh_library") {
    fun sceMpegInit(): Int {
        return -1 // @TODO: 0 when we implement the rest of the API
    }

    fun sceMpegQueryUserdataEsSize(cpu: CpuState): Unit = UNIMPLEMENTED(0x01977054)
    fun sceMpeg_0558B075(cpu: CpuState): Unit = UNIMPLEMENTED(0x0558B075)
    fun sceMpegAvcDecode(cpu: CpuState): Unit = UNIMPLEMENTED(0x0E3C2E9D)
    fun sceMpegAvcDecodeDetail(cpu: CpuState): Unit = UNIMPLEMENTED(0x0F6C18D7)
    fun sceMpeg_11CAB459(cpu: CpuState): Unit = UNIMPLEMENTED(0x11CAB459)
    fun sceMpegGetAvcNalAu(cpu: CpuState): Unit = UNIMPLEMENTED(0x11F95CF1)
    fun sceMpegRingbufferDestruct(cpu: CpuState): Unit = UNIMPLEMENTED(0x13407F13)
    fun sceMpegInitAu(cpu: CpuState): Unit = UNIMPLEMENTED(0x167AFD9E)
    fun sceMpegAvcQueryYCbCrSize(cpu: CpuState): Unit = UNIMPLEMENTED(0x211A057C)
    fun sceMpegQueryStreamOffset(cpu: CpuState): Unit = UNIMPLEMENTED(0x21FF80E4)
    fun sceMpegChangeGetAvcAuMode(cpu: CpuState): Unit = UNIMPLEMENTED(0x234586AE)
    fun sceMpegAvcCsc(cpu: CpuState): Unit = UNIMPLEMENTED(0x31BD0272)
    fun sceMpegRingbufferConstruct(cpu: CpuState): Unit = UNIMPLEMENTED(0x37295ED8)
    fun sceMpegNextAvcRpAu(cpu: CpuState): Unit = UNIMPLEMENTED(0x3C37A7A6)
    fun sceMpegRegistStream(cpu: CpuState): Unit = UNIMPLEMENTED(0x42560F23)
    fun sceMpeg_42C679F6(cpu: CpuState): Unit = UNIMPLEMENTED(0x42C679F6)
    fun sceMpegAvcDecodeFlush(cpu: CpuState): Unit = UNIMPLEMENTED(0x4571CC64)
    fun sceMpegFlushStream(cpu: CpuState): Unit = UNIMPLEMENTED(0x500F0429)
    fun sceMpegUnRegistStream(cpu: CpuState): Unit = UNIMPLEMENTED(0x591A4AA2)
    fun sceMpeg_5AC68A41(cpu: CpuState): Unit = UNIMPLEMENTED(0x5AC68A41)
    fun sceMpegDelete(cpu: CpuState): Unit = UNIMPLEMENTED(0x606A4649)
    fun sceMpegQueryStreamSize(cpu: CpuState): Unit = UNIMPLEMENTED(0x611E9E11)
    fun sceMpegAvcInitYCbCr(cpu: CpuState): Unit = UNIMPLEMENTED(0x67179B1B)
    fun sceMpegAvcDecodeGetDecodeSEI(cpu: CpuState): Unit = UNIMPLEMENTED(0x6F314410)
    fun sceMpegFlushAllStream(cpu: CpuState): Unit = UNIMPLEMENTED(0x707B7629)
    fun sceMpegAvcDecodeStop(cpu: CpuState): Unit = UNIMPLEMENTED(0x740FCCD1)
    fun sceMpeg_75E21135(cpu: CpuState): Unit = UNIMPLEMENTED(0x75E21135)
    fun sceMpegRingbufferQueryPackNum(cpu: CpuState): Unit = UNIMPLEMENTED(0x769BEBB6)
    fun sceMpegAtracDecode(cpu: CpuState): Unit = UNIMPLEMENTED(0x800C44DF)
    fun sceMpegFinish(cpu: CpuState): Unit = UNIMPLEMENTED(0x874624D6)
    fun sceMpegGetPcmAu(cpu: CpuState): Unit = UNIMPLEMENTED(0x8C1E027D)
    fun sceMpegGetAvcEsAu(cpu: CpuState): Unit = UNIMPLEMENTED(0x921FCCCF)
    fun sceMpeg_988E9E12(cpu: CpuState): Unit = UNIMPLEMENTED(0x988E9E12)
    fun sceMpegChangeGetAuMode(cpu: CpuState): Unit = UNIMPLEMENTED(0x9DCFB7EA)
    fun sceMpegAvcDecodeMode(cpu: CpuState): Unit = UNIMPLEMENTED(0xA11C7026)
    fun sceMpegMallocAvcEsBuf(cpu: CpuState): Unit = UNIMPLEMENTED(0xA780CF7E)
    fun sceMpeg_AB0E9556(cpu: CpuState): Unit = UNIMPLEMENTED(0xAB0E9556)
    fun sceMpeg_AE693D0B(cpu: CpuState): Unit = UNIMPLEMENTED(0xAE693D0B)
    fun sceMpegRingbufferPut(cpu: CpuState): Unit = UNIMPLEMENTED(0xB240A59E)
    fun sceMpeg_B27711A8(cpu: CpuState): Unit = UNIMPLEMENTED(0xB27711A8)
    fun sceMpegRingbufferAvailableSize(cpu: CpuState): Unit = UNIMPLEMENTED(0xB5F6DC87)
    fun sceMpegQueryPcmEsSize(cpu: CpuState): Unit = UNIMPLEMENTED(0xC02CF6B5)
    fun sceMpegQueryMemSize(cpu: CpuState): Unit = UNIMPLEMENTED(0xC132E22F)
    fun sceMpeg_C2F02CDD(cpu: CpuState): Unit = UNIMPLEMENTED(0xC2F02CDD)
    fun sceMpeg_C345DED2(cpu: CpuState): Unit = UNIMPLEMENTED(0xC345DED2)
    fun sceMpegFreeAvcEsBuf(cpu: CpuState): Unit = UNIMPLEMENTED(0xCEB870B1)
    fun sceMpegAvcDecodeDetail2(cpu: CpuState): Unit = UNIMPLEMENTED(0xCF3547A2)
    fun sceMpegAvcCscMode(cpu: CpuState): Unit = UNIMPLEMENTED(0xD1CE4950)
    fun sceMpeg_D4DD6E75(cpu: CpuState): Unit = UNIMPLEMENTED(0xD4DD6E75)
    fun sceMpegRingbufferQueryMemSize(cpu: CpuState): Unit = UNIMPLEMENTED(0xD7A29F46)
    fun sceMpegCreate(cpu: CpuState): Unit = UNIMPLEMENTED(0xD8C5F121)
    fun sceMpegFlushAu(cpu: CpuState): Unit = UNIMPLEMENTED(0xDBB60658)
    fun sceMpegGetAtracAu(cpu: CpuState): Unit = UNIMPLEMENTED(0xE1CE83A7)
    fun sceMpeg_E49EB257(cpu: CpuState): Unit = UNIMPLEMENTED(0xE49EB257)
    fun sceMpegAvcCscInfo(cpu: CpuState): Unit = UNIMPLEMENTED(0xE95838F6)
    fun sceMpegAvcDecodeYCbCr(cpu: CpuState): Unit = UNIMPLEMENTED(0xF0EB1125)
    fun sceMpegAvcDecodeStopYCbCr(cpu: CpuState): Unit = UNIMPLEMENTED(0xF2930C9C)
    fun sceMpegQueryAtracEsSize(cpu: CpuState): Unit = UNIMPLEMENTED(0xF8DCB679)
    fun sceMpegGetAvcAu(cpu: CpuState): Unit = UNIMPLEMENTED(0xFE246728)


    override fun registerModule() {
        registerFunctionInt("sceMpegInit", 0x682A619B, since = 150) { sceMpegInit() }

        registerFunctionRaw("sceMpegQueryUserdataEsSize", 0x01977054, since = 150) { sceMpegQueryUserdataEsSize(it) }
        registerFunctionRaw("sceMpeg_0558B075", 0x0558B075, since = 150) { sceMpeg_0558B075(it) }
        registerFunctionRaw("sceMpegAvcDecode", 0x0E3C2E9D, since = 150) { sceMpegAvcDecode(it) }
        registerFunctionRaw("sceMpegAvcDecodeDetail", 0x0F6C18D7, since = 150) { sceMpegAvcDecodeDetail(it) }
        registerFunctionRaw("sceMpeg_11CAB459", 0x11CAB459, since = 150) { sceMpeg_11CAB459(it) }
        registerFunctionRaw("sceMpegGetAvcNalAu", 0x11F95CF1, since = 150) { sceMpegGetAvcNalAu(it) }
        registerFunctionRaw("sceMpegRingbufferDestruct", 0x13407F13, since = 150) { sceMpegRingbufferDestruct(it) }
        registerFunctionRaw("sceMpegInitAu", 0x167AFD9E, since = 150) { sceMpegInitAu(it) }
        registerFunctionRaw("sceMpegAvcQueryYCbCrSize", 0x211A057C, since = 150) { sceMpegAvcQueryYCbCrSize(it) }
        registerFunctionRaw("sceMpegQueryStreamOffset", 0x21FF80E4, since = 150) { sceMpegQueryStreamOffset(it) }
        registerFunctionRaw("sceMpegChangeGetAvcAuMode", 0x234586AE, since = 150) { sceMpegChangeGetAvcAuMode(it) }
        registerFunctionRaw("sceMpegAvcCsc", 0x31BD0272, since = 150) { sceMpegAvcCsc(it) }
        registerFunctionRaw("sceMpegRingbufferConstruct", 0x37295ED8, since = 150) { sceMpegRingbufferConstruct(it) }
        registerFunctionRaw("sceMpegNextAvcRpAu", 0x3C37A7A6, since = 150) { sceMpegNextAvcRpAu(it) }
        registerFunctionRaw("sceMpegRegistStream", 0x42560F23, since = 150) { sceMpegRegistStream(it) }
        registerFunctionRaw("sceMpeg_42C679F6", 0x42C679F6, since = 150) { sceMpeg_42C679F6(it) }
        registerFunctionRaw("sceMpegAvcDecodeFlush", 0x4571CC64, since = 150) { sceMpegAvcDecodeFlush(it) }
        registerFunctionRaw("sceMpegFlushStream", 0x500F0429, since = 150) { sceMpegFlushStream(it) }
        registerFunctionRaw("sceMpegUnRegistStream", 0x591A4AA2, since = 150) { sceMpegUnRegistStream(it) }
        registerFunctionRaw("sceMpeg_5AC68A41", 0x5AC68A41, since = 150) { sceMpeg_5AC68A41(it) }
        registerFunctionRaw("sceMpegDelete", 0x606A4649, since = 150) { sceMpegDelete(it) }
        registerFunctionRaw("sceMpegQueryStreamSize", 0x611E9E11, since = 150) { sceMpegQueryStreamSize(it) }
        registerFunctionRaw("sceMpegAvcInitYCbCr", 0x67179B1B, since = 150) { sceMpegAvcInitYCbCr(it) }
        registerFunctionRaw(
            "sceMpegAvcDecodeGetDecodeSEI",
            0x6F314410,
            since = 150
        ) { sceMpegAvcDecodeGetDecodeSEI(it) }
        registerFunctionRaw("sceMpegFlushAllStream", 0x707B7629, since = 150) { sceMpegFlushAllStream(it) }
        registerFunctionRaw("sceMpegAvcDecodeStop", 0x740FCCD1, since = 150) { sceMpegAvcDecodeStop(it) }
        registerFunctionRaw("sceMpeg_75E21135", 0x75E21135, since = 150) { sceMpeg_75E21135(it) }
        registerFunctionRaw(
            "sceMpegRingbufferQueryPackNum",
            0x769BEBB6,
            since = 150
        ) { sceMpegRingbufferQueryPackNum(it) }
        registerFunctionRaw("sceMpegAtracDecode", 0x800C44DF, since = 150) { sceMpegAtracDecode(it) }
        registerFunctionRaw("sceMpegFinish", 0x874624D6, since = 150) { sceMpegFinish(it) }
        registerFunctionRaw("sceMpegGetPcmAu", 0x8C1E027D, since = 150) { sceMpegGetPcmAu(it) }
        registerFunctionRaw("sceMpegGetAvcEsAu", 0x921FCCCF, since = 150) { sceMpegGetAvcEsAu(it) }
        registerFunctionRaw("sceMpeg_988E9E12", 0x988E9E12, since = 150) { sceMpeg_988E9E12(it) }
        registerFunctionRaw("sceMpegChangeGetAuMode", 0x9DCFB7EA, since = 150) { sceMpegChangeGetAuMode(it) }
        registerFunctionRaw("sceMpegAvcDecodeMode", 0xA11C7026, since = 150) { sceMpegAvcDecodeMode(it) }
        registerFunctionRaw("sceMpegMallocAvcEsBuf", 0xA780CF7E, since = 150) { sceMpegMallocAvcEsBuf(it) }
        registerFunctionRaw("sceMpeg_AB0E9556", 0xAB0E9556, since = 150) { sceMpeg_AB0E9556(it) }
        registerFunctionRaw("sceMpeg_AE693D0B", 0xAE693D0B, since = 150) { sceMpeg_AE693D0B(it) }
        registerFunctionRaw("sceMpegRingbufferPut", 0xB240A59E, since = 150) { sceMpegRingbufferPut(it) }
        registerFunctionRaw("sceMpeg_B27711A8", 0xB27711A8, since = 150) { sceMpeg_B27711A8(it) }
        registerFunctionRaw("sceMpegRingbufferAvailableSize", 0xB5F6DC87, since = 150) {
            sceMpegRingbufferAvailableSize(
                it
            )
        }
        registerFunctionRaw("sceMpegQueryPcmEsSize", 0xC02CF6B5, since = 150) { sceMpegQueryPcmEsSize(it) }
        registerFunctionRaw("sceMpegQueryMemSize", 0xC132E22F, since = 150) { sceMpegQueryMemSize(it) }
        registerFunctionRaw("sceMpeg_C2F02CDD", 0xC2F02CDD, since = 150) { sceMpeg_C2F02CDD(it) }
        registerFunctionRaw("sceMpeg_C345DED2", 0xC345DED2, since = 150) { sceMpeg_C345DED2(it) }
        registerFunctionRaw("sceMpegQueryUserdataEsSize", 0xC45C99CC, since = 150) { sceMpegQueryUserdataEsSize(it) }
        registerFunctionRaw("sceMpegFreeAvcEsBuf", 0xCEB870B1, since = 150) { sceMpegFreeAvcEsBuf(it) }
        registerFunctionRaw("sceMpegAvcDecodeDetail2", 0xCF3547A2, since = 150) { sceMpegAvcDecodeDetail2(it) }
        registerFunctionRaw("sceMpegAvcCscMode", 0xD1CE4950, since = 150) { sceMpegAvcCscMode(it) }
        registerFunctionRaw("sceMpeg_D4DD6E75", 0xD4DD6E75, since = 150) { sceMpeg_D4DD6E75(it) }
        registerFunctionRaw(
            "sceMpegRingbufferQueryMemSize",
            0xD7A29F46,
            since = 150
        ) { sceMpegRingbufferQueryMemSize(it) }
        registerFunctionRaw("sceMpegCreate", 0xD8C5F121, since = 150) { sceMpegCreate(it) }
        registerFunctionRaw("sceMpegFlushAu", 0xDBB60658, since = 150) { sceMpegFlushAu(it) }
        registerFunctionRaw("sceMpegGetAtracAu", 0xE1CE83A7, since = 150) { sceMpegGetAtracAu(it) }
        registerFunctionRaw("sceMpeg_E49EB257", 0xE49EB257, since = 150) { sceMpeg_E49EB257(it) }
        registerFunctionRaw("sceMpegAvcCscInfo", 0xE95838F6, since = 150) { sceMpegAvcCscInfo(it) }
        registerFunctionRaw("sceMpegAvcDecodeYCbCr", 0xF0EB1125, since = 150) { sceMpegAvcDecodeYCbCr(it) }
        registerFunctionRaw("sceMpegAvcDecodeStopYCbCr", 0xF2930C9C, since = 150) { sceMpegAvcDecodeStopYCbCr(it) }
        registerFunctionRaw("sceMpegQueryAtracEsSize", 0xF8DCB679, since = 150) { sceMpegQueryAtracEsSize(it) }
        registerFunctionRaw("sceMpegGetAvcAu", 0xFE246728, since = 150) { sceMpegGetAvcAu(it) }
    }
}
