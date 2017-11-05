package com.soywiz.kpspemu.hle.modules

import com.soywiz.kpspemu.Emulator
import com.soywiz.kpspemu.cpu.CpuState
import com.soywiz.kpspemu.hle.SceModule

@Suppress("UNUSED_PARAMETER")
class sceAtrac3plus(emulator: Emulator) : SceModule(emulator, "sceAtrac3plus", 0x00010011, "libatrac3plus.prx", "sceATRAC3plus_Library") {
	fun sceAtracSetData(cpu: CpuState): Unit = UNIMPLEMENTED(0x0E2A73AB)
	fun sceAtracSetHalfwayBufferAndGetID(cpu: CpuState): Unit = UNIMPLEMENTED(0x0FAE370E)
	fun sceAtracReinit(cpu: CpuState): Unit = UNIMPLEMENTED(0x132F1ECA)
	fun sceAtrac3plus_2DD3E298(cpu: CpuState): Unit = UNIMPLEMENTED(0x2DD3E298)
	fun sceAtracGetChannel(cpu: CpuState): Unit = UNIMPLEMENTED(0x31668BAA)
	fun sceAtracGetNextSample(cpu: CpuState): Unit = UNIMPLEMENTED(0x36FAABFB)
	fun sceAtracSetHalfwayBuffer(cpu: CpuState): Unit = UNIMPLEMENTED(0x3F6E26B5)
	fun sceAtracSetAA3DataAndGetID(cpu: CpuState): Unit = UNIMPLEMENTED(0x5622B7C1)
	fun sceAtracSetMOutHalfwayBuffer(cpu: CpuState): Unit = UNIMPLEMENTED(0x5CF9D852)
	fun sceAtracGetStreamDataInfo(cpu: CpuState): Unit = UNIMPLEMENTED(0x5D268707)
	fun sceAtracSetAA3HalfwayBufferAndGetID(cpu: CpuState): Unit = UNIMPLEMENTED(0x5DD66588)
	fun sceAtracReleaseAtracID(cpu: CpuState): Unit = UNIMPLEMENTED(0x61EB33F5)
	fun sceAtracResetPlayPosition(cpu: CpuState): Unit = UNIMPLEMENTED(0x644E5607)
	fun sceAtracDecodeData(cpu: CpuState): Unit = UNIMPLEMENTED(0x6A8C3CD5)
	fun sceAtracGetAtracID(cpu: CpuState): Unit = UNIMPLEMENTED(0x780F88D1)
	fun sceAtracSetDataAndGetID(cpu: CpuState): Unit = UNIMPLEMENTED(0x7A20E7AF)
	fun sceAtracAddStreamData(cpu: CpuState): Unit = UNIMPLEMENTED(0x7DB31251)
	fun sceAtracSetSecondBuffer(cpu: CpuState): Unit = UNIMPLEMENTED(0x83BF7AFD)
	fun sceAtracGetSecondBufferInfo(cpu: CpuState): Unit = UNIMPLEMENTED(0x83E85EA0)
	fun sceAtracSetLoopNum(cpu: CpuState): Unit = UNIMPLEMENTED(0x868120B5)
	fun sceAtracGetRemainFrame(cpu: CpuState): Unit = UNIMPLEMENTED(0x9AE849A7)
	fun sceAtracSetMOutHalfwayBufferAndGetID(cpu: CpuState): Unit = UNIMPLEMENTED(0x9CD7DE03)
	fun sceAtracGetSoundSample(cpu: CpuState): Unit = UNIMPLEMENTED(0xA2BBA8BE)
	fun sceAtracGetBitrate(cpu: CpuState): Unit = UNIMPLEMENTED(0xA554A158)
	fun sceAtracGetOutputChannel(cpu: CpuState): Unit = UNIMPLEMENTED(0xB3B5D042)
	fun sceAtracGetBufferInfoForReseting(cpu: CpuState): Unit = UNIMPLEMENTED(0xCA3CA3D2)
	fun sceAtracStartEntry(cpu: CpuState): Unit = UNIMPLEMENTED(0xD1F59FDB)
	fun sceAtracEndEntry(cpu: CpuState): Unit = UNIMPLEMENTED(0xD5C28CC0)
	fun sceAtracGetMaxSample(cpu: CpuState): Unit = UNIMPLEMENTED(0xD6A5F2F7)
	fun sceAtracGetNextDecodePosition(cpu: CpuState): Unit = UNIMPLEMENTED(0xE23E3A35)
	fun sceAtracGetInternalErrorInfo(cpu: CpuState): Unit = UNIMPLEMENTED(0xE88F759B)
	fun sceAtracIsSecondBufferNeeded(cpu: CpuState): Unit = UNIMPLEMENTED(0xECA32A99)
	fun sceAtracGetLoopStatus(cpu: CpuState): Unit = UNIMPLEMENTED(0xFAA4F89B)


	override fun registerModule() {
		registerFunctionRaw("sceAtracSetData", 0x0E2A73AB, since = 150) { sceAtracSetData(it) }
		registerFunctionRaw("sceAtracSetHalfwayBufferAndGetID", 0x0FAE370E, since = 150) { sceAtracSetHalfwayBufferAndGetID(it) }
		registerFunctionRaw("sceAtracReinit", 0x132F1ECA, since = 150) { sceAtracReinit(it) }
		registerFunctionRaw("sceAtrac3plus_2DD3E298", 0x2DD3E298, since = 150) { sceAtrac3plus_2DD3E298(it) }
		registerFunctionRaw("sceAtracGetChannel", 0x31668BAA, since = 150) { sceAtracGetChannel(it) }
		registerFunctionRaw("sceAtracGetNextSample", 0x36FAABFB, since = 150) { sceAtracGetNextSample(it) }
		registerFunctionRaw("sceAtracSetHalfwayBuffer", 0x3F6E26B5, since = 150) { sceAtracSetHalfwayBuffer(it) }
		registerFunctionRaw("sceAtracSetAA3DataAndGetID", 0x5622B7C1, since = 150) { sceAtracSetAA3DataAndGetID(it) }
		registerFunctionRaw("sceAtracSetMOutHalfwayBuffer", 0x5CF9D852, since = 150) { sceAtracSetMOutHalfwayBuffer(it) }
		registerFunctionRaw("sceAtracGetStreamDataInfo", 0x5D268707, since = 150) { sceAtracGetStreamDataInfo(it) }
		registerFunctionRaw("sceAtracSetAA3HalfwayBufferAndGetID", 0x5DD66588, since = 150) { sceAtracSetAA3HalfwayBufferAndGetID(it) }
		registerFunctionRaw("sceAtracReleaseAtracID", 0x61EB33F5, since = 150) { sceAtracReleaseAtracID(it) }
		registerFunctionRaw("sceAtracResetPlayPosition", 0x644E5607, since = 150) { sceAtracResetPlayPosition(it) }
		registerFunctionRaw("sceAtracDecodeData", 0x6A8C3CD5, since = 150) { sceAtracDecodeData(it) }
		registerFunctionRaw("sceAtracGetAtracID", 0x780F88D1, since = 150) { sceAtracGetAtracID(it) }
		registerFunctionRaw("sceAtracSetDataAndGetID", 0x7A20E7AF, since = 150) { sceAtracSetDataAndGetID(it) }
		registerFunctionRaw("sceAtracAddStreamData", 0x7DB31251, since = 150) { sceAtracAddStreamData(it) }
		registerFunctionRaw("sceAtracSetSecondBuffer", 0x83BF7AFD, since = 150) { sceAtracSetSecondBuffer(it) }
		registerFunctionRaw("sceAtracGetSecondBufferInfo", 0x83E85EA0, since = 150) { sceAtracGetSecondBufferInfo(it) }
		registerFunctionRaw("sceAtracSetLoopNum", 0x868120B5, since = 150) { sceAtracSetLoopNum(it) }
		registerFunctionRaw("sceAtracGetRemainFrame", 0x9AE849A7, since = 150) { sceAtracGetRemainFrame(it) }
		registerFunctionRaw("sceAtracSetMOutHalfwayBufferAndGetID", 0x9CD7DE03, since = 150) { sceAtracSetMOutHalfwayBufferAndGetID(it) }
		registerFunctionRaw("sceAtracGetSoundSample", 0xA2BBA8BE, since = 150) { sceAtracGetSoundSample(it) }
		registerFunctionRaw("sceAtracGetBitrate", 0xA554A158, since = 150) { sceAtracGetBitrate(it) }
		registerFunctionRaw("sceAtracGetOutputChannel", 0xB3B5D042, since = 150) { sceAtracGetOutputChannel(it) }
		registerFunctionRaw("sceAtracGetBufferInfoForReseting", 0xCA3CA3D2, since = 150) { sceAtracGetBufferInfoForReseting(it) }
		registerFunctionRaw("sceAtracStartEntry", 0xD1F59FDB, since = 150) { sceAtracStartEntry(it) }
		registerFunctionRaw("sceAtracEndEntry", 0xD5C28CC0, since = 150) { sceAtracEndEntry(it) }
		registerFunctionRaw("sceAtracGetMaxSample", 0xD6A5F2F7, since = 150) { sceAtracGetMaxSample(it) }
		registerFunctionRaw("sceAtracGetNextDecodePosition", 0xE23E3A35, since = 150) { sceAtracGetNextDecodePosition(it) }
		registerFunctionRaw("sceAtracGetInternalErrorInfo", 0xE88F759B, since = 150) { sceAtracGetInternalErrorInfo(it) }
		registerFunctionRaw("sceAtracIsSecondBufferNeeded", 0xECA32A99, since = 150) { sceAtracIsSecondBufferNeeded(it) }
		registerFunctionRaw("sceAtracGetLoopStatus", 0xFAA4F89B, since = 150) { sceAtracGetLoopStatus(it) }
	}
}
