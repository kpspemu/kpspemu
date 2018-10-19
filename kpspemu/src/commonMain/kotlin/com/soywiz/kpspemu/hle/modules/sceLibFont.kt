package com.soywiz.kpspemu.hle.modules

import com.soywiz.kpspemu.*
import com.soywiz.kpspemu.cpu.*
import com.soywiz.kpspemu.hle.*

@Suppress("UNUSED_PARAMETER")
class sceLibFont(emulator: Emulator) :
    SceModule(emulator, "sceLibFont", 0x00010000, "libfont_hv.prx", "sceFont_Library_HV") {
    fun sceFontFlush(cpu: CpuState): Unit = UNIMPLEMENTED(0x02D7F94B)
    fun sceFontFindOptimumFont(cpu: CpuState): Unit = UNIMPLEMENTED(0x099EF33C)
    fun sceFontGetFontInfo(cpu: CpuState): Unit = UNIMPLEMENTED(0x0DA7535E)
    fun sceLibFont_HV_26149723(cpu: CpuState): Unit = UNIMPLEMENTED(0x26149723)
    fun sceFontGetNumFontList(cpu: CpuState): Unit = UNIMPLEMENTED(0x27F6E642)
    fun sceFontCalcMemorySize(cpu: CpuState): Unit = UNIMPLEMENTED(0x2F67356A)
    fun sceLibFont_HV_33FFD07C(cpu: CpuState): Unit = UNIMPLEMENTED(0x33FFD07C)
    fun sceFontClose(cpu: CpuState): Unit = UNIMPLEMENTED(0x3AEA8CB6)
    fun sceFontPointToPixelV(cpu: CpuState): Unit = UNIMPLEMENTED(0x3C4B7E82)
    fun sceFontPointToPixelH(cpu: CpuState): Unit = UNIMPLEMENTED(0x472694CD)
    fun sceFontSetResolution(cpu: CpuState): Unit = UNIMPLEMENTED(0x48293280)
    fun sceLibFont_HV_48592C48(cpu: CpuState): Unit = UNIMPLEMENTED(0x48592C48)
    fun sceFontGetShadowImageRect(cpu: CpuState): Unit = UNIMPLEMENTED(0x48B06520)
    fun sceFontGetFontInfoByIndexNumber(cpu: CpuState): Unit = UNIMPLEMENTED(0x5333322D)
    fun sceFontGetShadowGlyphImage(cpu: CpuState): Unit = UNIMPLEMENTED(0x568BE516)
    fun sceFontDoneLib(cpu: CpuState): Unit = UNIMPLEMENTED(0x574B6FBC)
    fun sceFontOpenUserFile(cpu: CpuState): Unit = UNIMPLEMENTED(0x57FCB733)
    fun sceFontGetCharImageRect(cpu: CpuState): Unit = UNIMPLEMENTED(0x5C3E4A9E)
    fun sceFontGetShadowGlyphImage_Clip(cpu: CpuState): Unit = UNIMPLEMENTED(0x5DCF6858)
    fun sceFontNewLib(cpu: CpuState): Unit = UNIMPLEMENTED(0x67F17ED7)
    fun sceFontFindFont(cpu: CpuState): Unit = UNIMPLEMENTED(0x681E61A7)
    fun sceFontPixelToPointH(cpu: CpuState): Unit = UNIMPLEMENTED(0x74B21701)
    fun sceFontGetCharGlyphImage(cpu: CpuState): Unit = UNIMPLEMENTED(0x980F4895)
    fun sceFontOpen(cpu: CpuState): Unit = UNIMPLEMENTED(0xA834319D)
    fun sceFontGetShadowInfo(cpu: CpuState): Unit = UNIMPLEMENTED(0xAA3DE7B5)
    fun sceFontOpenUserMemory(cpu: CpuState): Unit = UNIMPLEMENTED(0xBB8E7FE6)
    fun sceFontGetFontList(cpu: CpuState): Unit = UNIMPLEMENTED(0xBC75D85B)
    fun sceFontGetCharGlyphImage_Clip(cpu: CpuState): Unit = UNIMPLEMENTED(0xCA1E6945)
    fun sceFontGetCharInfo(cpu: CpuState): Unit = UNIMPLEMENTED(0xDCC80C2F)
    fun sceLibFont_HV_E0DBDE75(cpu: CpuState): Unit = UNIMPLEMENTED(0xE0DBDE75)
    fun sceLibFont_HV_E4606649(cpu: CpuState): Unit = UNIMPLEMENTED(0xE4606649)
    fun sceFontSetAltCharacterCode(cpu: CpuState): Unit = UNIMPLEMENTED(0xEE232411)
    fun sceLibFont_HV_F0AD0FE7(cpu: CpuState): Unit = UNIMPLEMENTED(0xF0AD0FE7)
    fun sceFontPixelToPointV(cpu: CpuState): Unit = UNIMPLEMENTED(0xF8F0752E)


    override fun registerModule() {
        registerFunctionRaw("sceFontFlush", 0x02D7F94B, since = 150) { sceFontFlush(it) }
        registerFunctionRaw("sceFontFindOptimumFont", 0x099EF33C, since = 150) { sceFontFindOptimumFont(it) }
        registerFunctionRaw("sceFontGetFontInfo", 0x0DA7535E, since = 150) { sceFontGetFontInfo(it) }
        registerFunctionRaw("sceLibFont_HV_26149723", 0x26149723, since = 150) { sceLibFont_HV_26149723(it) }
        registerFunctionRaw("sceFontGetNumFontList", 0x27F6E642, since = 150) { sceFontGetNumFontList(it) }
        registerFunctionRaw("sceFontCalcMemorySize", 0x2F67356A, since = 150) { sceFontCalcMemorySize(it) }
        registerFunctionRaw("sceLibFont_HV_33FFD07C", 0x33FFD07C, since = 150) { sceLibFont_HV_33FFD07C(it) }
        registerFunctionRaw("sceFontClose", 0x3AEA8CB6, since = 150) { sceFontClose(it) }
        registerFunctionRaw("sceFontPointToPixelV", 0x3C4B7E82, since = 150) { sceFontPointToPixelV(it) }
        registerFunctionRaw("sceFontPointToPixelH", 0x472694CD, since = 150) { sceFontPointToPixelH(it) }
        registerFunctionRaw("sceFontSetResolution", 0x48293280, since = 150) { sceFontSetResolution(it) }
        registerFunctionRaw("sceLibFont_HV_48592C48", 0x48592C48, since = 150) { sceLibFont_HV_48592C48(it) }
        registerFunctionRaw("sceFontGetShadowImageRect", 0x48B06520, since = 150) { sceFontGetShadowImageRect(it) }
        registerFunctionRaw(
            "sceFontGetFontInfoByIndexNumber",
            0x5333322D,
            since = 150
        ) { sceFontGetFontInfoByIndexNumber(it) }
        registerFunctionRaw("sceFontGetShadowGlyphImage", 0x568BE516, since = 150) { sceFontGetShadowGlyphImage(it) }
        registerFunctionRaw("sceFontDoneLib", 0x574B6FBC, since = 150) { sceFontDoneLib(it) }
        registerFunctionRaw("sceFontOpenUserFile", 0x57FCB733, since = 150) { sceFontOpenUserFile(it) }
        registerFunctionRaw("sceFontGetCharImageRect", 0x5C3E4A9E, since = 150) { sceFontGetCharImageRect(it) }
        registerFunctionRaw(
            "sceFontGetShadowGlyphImage_Clip",
            0x5DCF6858,
            since = 150
        ) { sceFontGetShadowGlyphImage_Clip(it) }
        registerFunctionRaw("sceFontNewLib", 0x67F17ED7, since = 150) { sceFontNewLib(it) }
        registerFunctionRaw("sceFontFindFont", 0x681E61A7, since = 150) { sceFontFindFont(it) }
        registerFunctionRaw("sceFontPixelToPointH", 0x74B21701, since = 150) { sceFontPixelToPointH(it) }
        registerFunctionRaw("sceFontGetCharGlyphImage", 0x980F4895, since = 150) { sceFontGetCharGlyphImage(it) }
        registerFunctionRaw("sceFontOpen", 0xA834319D, since = 150) { sceFontOpen(it) }
        registerFunctionRaw("sceFontGetShadowInfo", 0xAA3DE7B5, since = 150) { sceFontGetShadowInfo(it) }
        registerFunctionRaw("sceFontOpenUserMemory", 0xBB8E7FE6, since = 150) { sceFontOpenUserMemory(it) }
        registerFunctionRaw("sceFontGetFontList", 0xBC75D85B, since = 150) { sceFontGetFontList(it) }
        registerFunctionRaw(
            "sceFontGetCharGlyphImage_Clip",
            0xCA1E6945,
            since = 150
        ) { sceFontGetCharGlyphImage_Clip(it) }
        registerFunctionRaw("sceFontGetCharInfo", 0xDCC80C2F, since = 150) { sceFontGetCharInfo(it) }
        registerFunctionRaw("sceLibFont_HV_E0DBDE75", 0xE0DBDE75, since = 150) { sceLibFont_HV_E0DBDE75(it) }
        registerFunctionRaw("sceLibFont_HV_E4606649", 0xE4606649, since = 150) { sceLibFont_HV_E4606649(it) }
        registerFunctionRaw("sceFontSetAltCharacterCode", 0xEE232411, since = 150) { sceFontSetAltCharacterCode(it) }
        registerFunctionRaw("sceLibFont_HV_F0AD0FE7", 0xF0AD0FE7, since = 150) { sceLibFont_HV_F0AD0FE7(it) }
        registerFunctionRaw("sceFontPixelToPointV", 0xF8F0752E, since = 150) { sceFontPixelToPointV(it) }
    }
}
