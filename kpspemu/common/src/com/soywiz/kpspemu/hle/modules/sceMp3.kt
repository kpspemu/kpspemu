package com.soywiz.kpspemu.hle.modules

import com.soywiz.kpspemu.*
import com.soywiz.kpspemu.cpu.*
import com.soywiz.kpspemu.hle.*

@Suppress("UNUSED_PARAMETER")
class sceMp3(emulator: Emulator) : SceModule(emulator, "sceMp3", 0x00010011, "libmp3.prx", "sceMp3_Library") {
    fun sceMp3ReserveMp3Handle(cpu: CpuState): Unit = UNIMPLEMENTED(0x07EC321A)
    fun sceMp3NotifyAddStreamData(cpu: CpuState): Unit = UNIMPLEMENTED(0x0DB149F4)
    fun sceMp3ResetPlayPosition(cpu: CpuState): Unit = UNIMPLEMENTED(0x2A368661)
    fun sceMp3GetSumDecodedSample(cpu: CpuState): Unit = UNIMPLEMENTED(0x354D27EA)
    fun sceMp3InitResource(cpu: CpuState): Unit = UNIMPLEMENTED(0x35750070)
    fun sceMp3TermResource(cpu: CpuState): Unit = UNIMPLEMENTED(0x3C2FA058)
    fun sceMp3SetLoopNum(cpu: CpuState): Unit = UNIMPLEMENTED(0x3CEF484F)
    fun sceMp3Init(cpu: CpuState): Unit = UNIMPLEMENTED(0x44E07129)
    fun sceMp3EndEntry(cpu: CpuState): Unit = UNIMPLEMENTED(0x732B042A)
    fun sceMp3GetMp3ChannelNum(cpu: CpuState): Unit = UNIMPLEMENTED(0x7F696782)
    fun sceMp3GetBitRate(cpu: CpuState): Unit = UNIMPLEMENTED(0x87677E40)
    fun sceMp3GetMaxOutputSample(cpu: CpuState): Unit = UNIMPLEMENTED(0x87C263D1)
    fun sceMp3StartEntry(cpu: CpuState): Unit = UNIMPLEMENTED(0x8AB81558)
    fun sceMp3GetSamplingRate(cpu: CpuState): Unit = UNIMPLEMENTED(0x8F450998)
    fun sceMp3GetInfoToAddStreamData(cpu: CpuState): Unit = UNIMPLEMENTED(0xA703FE0F)
    fun sceMp3Decode(cpu: CpuState): Unit = UNIMPLEMENTED(0xD021C0FB)
    fun sceMp3CheckStreamDataNeeded(cpu: CpuState): Unit = UNIMPLEMENTED(0xD0A56296)
    fun sceMp3GetLoopNum(cpu: CpuState): Unit = UNIMPLEMENTED(0xD8F54A51)
    fun sceMp3ReleaseMp3Handle(cpu: CpuState): Unit = UNIMPLEMENTED(0xF5478233)


    override fun registerModule() {
        registerFunctionRaw("sceMp3ReserveMp3Handle", 0x07EC321A, since = 150) { sceMp3ReserveMp3Handle(it) }
        registerFunctionRaw("sceMp3NotifyAddStreamData", 0x0DB149F4, since = 150) { sceMp3NotifyAddStreamData(it) }
        registerFunctionRaw("sceMp3ResetPlayPosition", 0x2A368661, since = 150) { sceMp3ResetPlayPosition(it) }
        registerFunctionRaw("sceMp3GetSumDecodedSample", 0x354D27EA, since = 150) { sceMp3GetSumDecodedSample(it) }
        registerFunctionRaw("sceMp3InitResource", 0x35750070, since = 150) { sceMp3InitResource(it) }
        registerFunctionRaw("sceMp3TermResource", 0x3C2FA058, since = 150) { sceMp3TermResource(it) }
        registerFunctionRaw("sceMp3SetLoopNum", 0x3CEF484F, since = 150) { sceMp3SetLoopNum(it) }
        registerFunctionRaw("sceMp3Init", 0x44E07129, since = 150) { sceMp3Init(it) }
        registerFunctionRaw("sceMp3EndEntry", 0x732B042A, since = 150) { sceMp3EndEntry(it) }
        registerFunctionRaw("sceMp3GetMp3ChannelNum", 0x7F696782, since = 150) { sceMp3GetMp3ChannelNum(it) }
        registerFunctionRaw("sceMp3GetBitRate", 0x87677E40, since = 150) { sceMp3GetBitRate(it) }
        registerFunctionRaw("sceMp3GetMaxOutputSample", 0x87C263D1, since = 150) { sceMp3GetMaxOutputSample(it) }
        registerFunctionRaw("sceMp3StartEntry", 0x8AB81558, since = 150) { sceMp3StartEntry(it) }
        registerFunctionRaw("sceMp3GetSamplingRate", 0x8F450998, since = 150) { sceMp3GetSamplingRate(it) }
        registerFunctionRaw(
            "sceMp3GetInfoToAddStreamData",
            0xA703FE0F,
            since = 150
        ) { sceMp3GetInfoToAddStreamData(it) }
        registerFunctionRaw("sceMp3Decode", 0xD021C0FB, since = 150) { sceMp3Decode(it) }
        registerFunctionRaw("sceMp3CheckStreamDataNeeded", 0xD0A56296, since = 150) { sceMp3CheckStreamDataNeeded(it) }
        registerFunctionRaw("sceMp3GetLoopNum", 0xD8F54A51, since = 150) { sceMp3GetLoopNum(it) }
        registerFunctionRaw("sceMp3ReleaseMp3Handle", 0xF5478233, since = 150) { sceMp3ReleaseMp3Handle(it) }
    }
}
