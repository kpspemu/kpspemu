package com.soywiz.kpspemu.hle.modules

import com.soywiz.kpspemu.*

@Suppress("UNUSED_PARAMETER")
class UtilsForKernel(emulator: Emulator) :
    UtilsBase(emulator, "UtilsForKernel", 0x00090011, "sysmem.prx", "sceSystemMemoryManager") {
    /*
    fun UtilsForKernel_004D4DEE(cpu: CpuState): Unit = UNIMPLEMENTED(0x004D4DEE)
    fun sceKernelUtilsMt19937UInt(cpu: CpuState): Unit = UNIMPLEMENTED(0x06FB8A63)
    fun sceKernelSetPTRIGMask(cpu: CpuState): Unit = UNIMPLEMENTED(0x136F2419)
    fun UtilsForKernel_157A383A(cpu: CpuState): Unit = UNIMPLEMENTED(0x157A383A)
    fun sceKernelSetGPIMask(cpu: CpuState): Unit = UNIMPLEMENTED(0x193D4036)
    fun UtilsForKernel_1B0592A3(cpu: CpuState): Unit = UNIMPLEMENTED(0x1B0592A3)
    fun sceKernelRegisterRtcFunc(cpu: CpuState): Unit = UNIMPLEMENTED(0x23A0C5BA)
    fun sceKernelGzipGetCompressedData(cpu: CpuState): Unit = UNIMPLEMENTED(0x23FFC828)
    fun sceKernelLibcTime(cpu: CpuState): Unit = UNIMPLEMENTED(0x27CC57F0)
    fun sceKernelUtilsSha1BlockUpdate(cpu: CpuState): Unit = UNIMPLEMENTED(0x346F6DA8)
    fun sceKernelGetGPI(cpu: CpuState): Unit = UNIMPLEMENTED(0x37FB5C42)
    fun sceKernelGetPTRIG(cpu: CpuState): Unit = UNIMPLEMENTED(0x39F49610)
    fun sceKernelDcacheWritebackRange(cpu: CpuState): Unit = UNIMPLEMENTED(0x3EE30821)
    fun UtilsForKernel_3FD3D324(cpu: CpuState): Unit = UNIMPLEMENTED(0x3FD3D324)
    fun sceKernelReleaseRtcFunc(cpu: CpuState): Unit = UNIMPLEMENTED(0x41887EF4)
    fun UtilsForKernel_43C9A8DB(cpu: CpuState): Unit = UNIMPLEMENTED(0x43C9A8DB)
    fun sceKernelIcacheProbe(cpu: CpuState): Unit = UNIMPLEMENTED(0x4FD31C9D)
    fun UtilsForKernel_515B4FAF(cpu: CpuState): Unit = UNIMPLEMENTED(0x515B4FAF)
    fun sceKernelUtilsSha1BlockResult(cpu: CpuState): Unit = UNIMPLEMENTED(0x585F1C09)
    fun UtilsForKernel_5C7F2B1A(cpu: CpuState): Unit = UNIMPLEMENTED(0x5C7F2B1A)
    fun sceKernelUtilsMd5BlockUpdate(cpu: CpuState): Unit = UNIMPLEMENTED(0x61E1E525)
    fun sceKernelSetPTRIG(cpu: CpuState): Unit = UNIMPLEMENTED(0x6231A71D)
    fun sceKernelSetGPO(cpu: CpuState): Unit = UNIMPLEMENTED(0x6AD345D7)
    fun UtilsForKernel_6C6887EE(cpu: CpuState): Unit = UNIMPLEMENTED(0x6C6887EE)
    fun sceKernelLibcGettimeofday(cpu: CpuState): Unit = UNIMPLEMENTED(0x71EC4271)
    fun UtilsForKernel_7333E539(cpu: CpuState): Unit = UNIMPLEMENTED(0x7333E539)
    fun UtilsForKernel_740DF7F0(cpu: CpuState): Unit = UNIMPLEMENTED(0x740DF7F0)
    fun sceKernelDcacheProbeRange(cpu: CpuState): Unit = UNIMPLEMENTED(0x77DFF087)
    fun sceKernelGzipDecompress(cpu: CpuState): Unit = UNIMPLEMENTED(0x78934841)
    fun sceKernelDcacheWritebackAll(cpu: CpuState): Unit = UNIMPLEMENTED(0x79D1C3FA)
    fun sceKernelDcacheProbe(cpu: CpuState): Unit = UNIMPLEMENTED(0x80001C4C)
    fun sceKernelUtilsSha1Digest(cpu: CpuState): Unit = UNIMPLEMENTED(0x840259F1)
    fun sceKernelDcacheInvalidateAll(cpu: CpuState): Unit = UNIMPLEMENTED(0x864A9D72)
    fun sceKernelPutUserLog(cpu: CpuState): Unit = UNIMPLEMENTED(0x87E81561)
    fun sceKernelGzipGetComment(cpu: CpuState): Unit = UNIMPLEMENTED(0x8C1FBE04)
    fun sceKernelIcacheInvalidateAll(cpu: CpuState): Unit = UNIMPLEMENTED(0x920F104A)
    fun sceKernelRegisterUserLogHandler(cpu: CpuState): Unit = UNIMPLEMENTED(0x92282A47)
    fun sceKernelSetGPOMask(cpu: CpuState): Unit = UNIMPLEMENTED(0x95035FEF)
    fun UtilsForKernel_99134C3F(cpu: CpuState): Unit = UNIMPLEMENTED(0x99134C3F)
    fun sceKernelUtilsMd5BlockInit(cpu: CpuState): Unit = UNIMPLEMENTED(0x9E5C5086)
    fun UtilsForKernel_AA9AF5CF(cpu: CpuState): Unit = UNIMPLEMENTED(0xAA9AF5CF)
    fun sceKernelGetGPO(cpu: CpuState): Unit = UNIMPLEMENTED(0xAF3616C0)
    fun UtilsForKernel_AF3766BB(cpu: CpuState): Unit = UNIMPLEMENTED(0xAF3766BB)
    fun sceKernelGzipGetInfo(cpu: CpuState): Unit = UNIMPLEMENTED(0xB0E9C31F)
    fun sceKernelDcacheWritebackInvalidateAll(cpu: CpuState): Unit = UNIMPLEMENTED(0xB435DEC5)
    fun UtilsForKernel_B83A1E76(cpu: CpuState): Unit = UNIMPLEMENTED(0xB83A1E76)
    fun sceKernelUtilsMd5BlockResult(cpu: CpuState): Unit = UNIMPLEMENTED(0xB8D24E78)
    fun sceKernelRtcGetTick(cpu: CpuState): Unit = UNIMPLEMENTED(0xBDBFCA89)
    fun sceKernelDcacheInvalidateRange(cpu: CpuState): Unit = UNIMPLEMENTED(0xBFA98062)
    fun sceKernelUtilsMd5Digest(cpu: CpuState): Unit = UNIMPLEMENTED(0xC8186A58)
    fun UtilsForKernel_DBBE9A46(cpu: CpuState): Unit = UNIMPLEMENTED(0xDBBE9A46)
    fun sceKernelGzipIsValid(cpu: CpuState): Unit = UNIMPLEMENTED(0xE0CE3E29)
    fun sceKernelGzipGetName(cpu: CpuState): Unit = UNIMPLEMENTED(0xE0E6BA96)
    fun sceKernelUtilsMt19937Init(cpu: CpuState): Unit = UNIMPLEMENTED(0xE860E75E)
    fun sceKernelDeflateDecompress(cpu: CpuState): Unit = UNIMPLEMENTED(0xE8DB3CE6)
    fun UtilsForKernel_F0155BCA(cpu: CpuState): Unit = UNIMPLEMENTED(0xF0155BCA)
    fun sceKernelUtilsSha1BlockInit(cpu: CpuState): Unit = UNIMPLEMENTED(0xF8FCD5BA)
    */

    override fun registerModule() {
        super.registerModule()
        /*
        registerFunctionVoid("sceKernelDcacheWritebackInvalidateRange", 0x34B9FA9E, since = 150) { sceKernelDcacheWritebackInvalidateRange(int, int) }
        registerFunctionInt("sceKernelLibcClock", 0x91E4F6A7, since = 150) { sceKernelLibcClock() }

        registerFunctionRaw("UtilsForKernel_004D4DEE", 0x004D4DEE, since = 150) { UtilsForKernel_004D4DEE(it) }
        registerFunctionRaw("sceKernelUtilsMt19937UInt", 0x06FB8A63, since = 150) { sceKernelUtilsMt19937UInt(it) }
        registerFunctionRaw("sceKernelSetPTRIGMask", 0x136F2419, since = 150) { sceKernelSetPTRIGMask(it) }
        registerFunctionRaw("UtilsForKernel_157A383A", 0x157A383A, since = 150) { UtilsForKernel_157A383A(it) }
        registerFunctionRaw("sceKernelSetGPIMask", 0x193D4036, since = 150) { sceKernelSetGPIMask(it) }
        registerFunctionRaw("UtilsForKernel_1B0592A3", 0x1B0592A3, since = 150) { UtilsForKernel_1B0592A3(it) }
        registerFunctionRaw("sceKernelRegisterRtcFunc", 0x23A0C5BA, since = 150) { sceKernelRegisterRtcFunc(it) }
        registerFunctionRaw("sceKernelGzipGetCompressedData", 0x23FFC828, since = 150) { sceKernelGzipGetCompressedData(it) }
        registerFunctionRaw("sceKernelLibcTime", 0x27CC57F0, since = 150) { sceKernelLibcTime(it) }
        registerFunctionRaw("sceKernelUtilsSha1BlockUpdate", 0x346F6DA8, since = 150) { sceKernelUtilsSha1BlockUpdate(it) }
        registerFunctionRaw("sceKernelGetGPI", 0x37FB5C42, since = 150) { sceKernelGetGPI(it) }
        registerFunctionRaw("sceKernelGetPTRIG", 0x39F49610, since = 150) { sceKernelGetPTRIG(it) }
        registerFunctionRaw("sceKernelDcacheWritebackRange", 0x3EE30821, since = 150) { sceKernelDcacheWritebackRange(it) }
        registerFunctionRaw("UtilsForKernel_3FD3D324", 0x3FD3D324, since = 150) { UtilsForKernel_3FD3D324(it) }
        registerFunctionRaw("sceKernelReleaseRtcFunc", 0x41887EF4, since = 150) { sceKernelReleaseRtcFunc(it) }
        registerFunctionRaw("UtilsForKernel_43C9A8DB", 0x43C9A8DB, since = 150) { UtilsForKernel_43C9A8DB(it) }
        registerFunctionRaw("sceKernelIcacheProbe", 0x4FD31C9D, since = 150) { sceKernelIcacheProbe(it) }
        registerFunctionRaw("UtilsForKernel_515B4FAF", 0x515B4FAF, since = 150) { UtilsForKernel_515B4FAF(it) }
        registerFunctionRaw("sceKernelUtilsSha1BlockResult", 0x585F1C09, since = 150) { sceKernelUtilsSha1BlockResult(it) }
        registerFunctionRaw("UtilsForKernel_5C7F2B1A", 0x5C7F2B1A, since = 150) { UtilsForKernel_5C7F2B1A(it) }
        registerFunctionRaw("sceKernelUtilsMd5BlockUpdate", 0x61E1E525, since = 150) { sceKernelUtilsMd5BlockUpdate(it) }
        registerFunctionRaw("sceKernelSetPTRIG", 0x6231A71D, since = 150) { sceKernelSetPTRIG(it) }
        registerFunctionRaw("sceKernelSetGPO", 0x6AD345D7, since = 150) { sceKernelSetGPO(it) }
        registerFunctionRaw("UtilsForKernel_6C6887EE", 0x6C6887EE, since = 150) { UtilsForKernel_6C6887EE(it) }
        registerFunctionRaw("sceKernelLibcGettimeofday", 0x71EC4271, since = 150) { sceKernelLibcGettimeofday(it) }
        registerFunctionRaw("UtilsForKernel_7333E539", 0x7333E539, since = 150) { UtilsForKernel_7333E539(it) }
        registerFunctionRaw("UtilsForKernel_740DF7F0", 0x740DF7F0, since = 150) { UtilsForKernel_740DF7F0(it) }
        registerFunctionRaw("sceKernelDcacheProbeRange", 0x77DFF087, since = 150) { sceKernelDcacheProbeRange(it) }
        registerFunctionRaw("sceKernelGzipDecompress", 0x78934841, since = 150) { sceKernelGzipDecompress(it) }
        registerFunctionRaw("sceKernelDcacheWritebackAll", 0x79D1C3FA, since = 150) { sceKernelDcacheWritebackAll(it) }
        registerFunctionRaw("sceKernelDcacheProbe", 0x80001C4C, since = 150) { sceKernelDcacheProbe(it) }
        registerFunctionRaw("sceKernelUtilsSha1Digest", 0x840259F1, since = 150) { sceKernelUtilsSha1Digest(it) }
        registerFunctionRaw("sceKernelDcacheInvalidateAll", 0x864A9D72, since = 150) { sceKernelDcacheInvalidateAll(it) }
        registerFunctionRaw("sceKernelPutUserLog", 0x87E81561, since = 150) { sceKernelPutUserLog(it) }
        registerFunctionRaw("sceKernelGzipGetComment", 0x8C1FBE04, since = 150) { sceKernelGzipGetComment(it) }
        registerFunctionRaw("sceKernelIcacheInvalidateAll", 0x920F104A, since = 150) { sceKernelIcacheInvalidateAll(it) }
        registerFunctionRaw("sceKernelRegisterUserLogHandler", 0x92282A47, since = 150) { sceKernelRegisterUserLogHandler(it) }
        registerFunctionRaw("sceKernelSetGPOMask", 0x95035FEF, since = 150) { sceKernelSetGPOMask(it) }
        registerFunctionRaw("UtilsForKernel_99134C3F", 0x99134C3F, since = 150) { UtilsForKernel_99134C3F(it) }
        registerFunctionRaw("sceKernelUtilsMd5BlockInit", 0x9E5C5086, since = 150) { sceKernelUtilsMd5BlockInit(it) }
        registerFunctionRaw("UtilsForKernel_AA9AF5CF", 0xAA9AF5CF, since = 150) { UtilsForKernel_AA9AF5CF(it) }
        registerFunctionRaw("sceKernelGetGPO", 0xAF3616C0, since = 150) { sceKernelGetGPO(it) }
        registerFunctionRaw("UtilsForKernel_AF3766BB", 0xAF3766BB, since = 150) { UtilsForKernel_AF3766BB(it) }
        registerFunctionRaw("sceKernelGzipGetInfo", 0xB0E9C31F, since = 150) { sceKernelGzipGetInfo(it) }
        registerFunctionRaw("sceKernelDcacheWritebackInvalidateAll", 0xB435DEC5, since = 150) { sceKernelDcacheWritebackInvalidateAll(it) }
        registerFunctionRaw("UtilsForKernel_B83A1E76", 0xB83A1E76, since = 150) { UtilsForKernel_B83A1E76(it) }
        registerFunctionRaw("sceKernelUtilsMd5BlockResult", 0xB8D24E78, since = 150) { sceKernelUtilsMd5BlockResult(it) }
        registerFunctionRaw("sceKernelRtcGetTick", 0xBDBFCA89, since = 150) { sceKernelRtcGetTick(it) }
        registerFunctionRaw("sceKernelDcacheInvalidateRange", 0xBFA98062, since = 150) { sceKernelDcacheInvalidateRange(it) }
        registerFunctionRaw("sceKernelUtilsMd5Digest", 0xC8186A58, since = 150) { sceKernelUtilsMd5Digest(it) }
        registerFunctionRaw("UtilsForKernel_DBBE9A46", 0xDBBE9A46, since = 150) { UtilsForKernel_DBBE9A46(it) }
        registerFunctionRaw("sceKernelGzipIsValid", 0xE0CE3E29, since = 150) { sceKernelGzipIsValid(it) }
        registerFunctionRaw("sceKernelGzipGetName", 0xE0E6BA96, since = 150) { sceKernelGzipGetName(it) }
        registerFunctionRaw("sceKernelUtilsMt19937Init", 0xE860E75E, since = 150) { sceKernelUtilsMt19937Init(it) }
        registerFunctionRaw("sceKernelDeflateDecompress", 0xE8DB3CE6, since = 150) { sceKernelDeflateDecompress(it) }
        registerFunctionRaw("UtilsForKernel_F0155BCA", 0xF0155BCA, since = 150) { UtilsForKernel_F0155BCA(it) }
        registerFunctionRaw("sceKernelUtilsSha1BlockInit", 0xF8FCD5BA, since = 150) { sceKernelUtilsSha1BlockInit(it) }
        */
    }
}
