package com.soywiz.kpspemu.hle.modules

import com.soywiz.kpspemu.Emulator
import com.soywiz.kpspemu.cpu.CpuState
import com.soywiz.kpspemu.hle.SceModule

@Suppress("UNUSED_PARAMETER")
class sceSasCore(emulator: Emulator) : SceModule(emulator, "sceSasCore", 0x40010011, "sc_sascore.prx", "sceSAScore") {
	fun __sceSasSetADSR(cpu: CpuState): Unit = UNIMPLEMENTED(0x019B25EB)
	fun __sceSasGetAllEnvelopeHeights(cpu: CpuState): Unit = UNIMPLEMENTED(0x07F58C24)
	fun __sceSasRevParam(cpu: CpuState): Unit = UNIMPLEMENTED(0x267A6DD2)
	fun __sceSasGetPauseFlag(cpu: CpuState): Unit = UNIMPLEMENTED(0x2C8E6AB3)
	fun __sceSasRevType(cpu: CpuState): Unit = UNIMPLEMENTED(0x33D4AB37)
	fun __sceSasInit(cpu: CpuState): Unit = UNIMPLEMENTED(0x42778A9F)
	fun __sceSasSetVolume(cpu: CpuState): Unit = UNIMPLEMENTED(0x440CA7D8)
	fun __sceSasCoreWithMix(cpu: CpuState): Unit = UNIMPLEMENTED(0x50A14DFC)
	fun __sceSasSetSL(cpu: CpuState): Unit = UNIMPLEMENTED(0x5F9529F6)
	fun __sceSasGetEndFlag(cpu: CpuState): Unit = UNIMPLEMENTED(0x68A46B95)
	fun __sceSasGetEnvelopeHeight(cpu: CpuState): Unit = UNIMPLEMENTED(0x74AE582A)
	fun __sceSasSetKeyOn(cpu: CpuState): Unit = UNIMPLEMENTED(0x76F01ACA)
	fun __sceSasSetPause(cpu: CpuState): Unit = UNIMPLEMENTED(0x787D04D5)
	fun __sceSasSetVoice(cpu: CpuState): Unit = UNIMPLEMENTED(0x99944089)
	fun __sceSasSetADSRmode(cpu: CpuState): Unit = UNIMPLEMENTED(0x9EC3676A)
	fun __sceSasSetKeyOff(cpu: CpuState): Unit = UNIMPLEMENTED(0xA0CF2FA4)
	fun __sceSasSetTrianglarWave(cpu: CpuState): Unit = UNIMPLEMENTED(0xA232CBE6)
	fun __sceSasCore(cpu: CpuState): Unit = UNIMPLEMENTED(0xA3589D81)
	fun __sceSasSetPitch(cpu: CpuState): Unit = UNIMPLEMENTED(0xAD84D37F)
	fun __sceSasSetNoise(cpu: CpuState): Unit = UNIMPLEMENTED(0xB7660A23)
	fun __sceSasGetGrain(cpu: CpuState): Unit = UNIMPLEMENTED(0xBD11B7C2)
	fun __sceSasSetSimpleADSR(cpu: CpuState): Unit = UNIMPLEMENTED(0xCBCD4F79)
	fun __sceSasSetGrain(cpu: CpuState): Unit = UNIMPLEMENTED(0xD1E0A01E)
	fun __sceSasRevEVOL(cpu: CpuState): Unit = UNIMPLEMENTED(0xD5A229C9)
	fun __sceSasSetSteepWave(cpu: CpuState): Unit = UNIMPLEMENTED(0xD5EBBBCD)
	fun __sceSasGetOutputmode(cpu: CpuState): Unit = UNIMPLEMENTED(0xE175EF66)
	fun __sceSasSetVoicePCM(cpu: CpuState): Unit = UNIMPLEMENTED(0xE1CD9561)
	fun __sceSasSetOutputmode(cpu: CpuState): Unit = UNIMPLEMENTED(0xE855BF76)
	fun __sceSasRevVON(cpu: CpuState): Unit = UNIMPLEMENTED(0xF983B186)


	override fun registerModule() {
		registerFunctionRaw("__sceSasSetADSR", 0x019B25EB, since = 150) { __sceSasSetADSR(it) }
		registerFunctionRaw("__sceSasGetAllEnvelopeHeights", 0x07F58C24, since = 150) { __sceSasGetAllEnvelopeHeights(it) }
		registerFunctionRaw("__sceSasRevParam", 0x267A6DD2, since = 150) { __sceSasRevParam(it) }
		registerFunctionRaw("__sceSasGetPauseFlag", 0x2C8E6AB3, since = 150) { __sceSasGetPauseFlag(it) }
		registerFunctionRaw("__sceSasRevType", 0x33D4AB37, since = 150) { __sceSasRevType(it) }
		registerFunctionRaw("__sceSasInit", 0x42778A9F, since = 150) { __sceSasInit(it) }
		registerFunctionRaw("__sceSasSetVolume", 0x440CA7D8, since = 150) { __sceSasSetVolume(it) }
		registerFunctionRaw("__sceSasCoreWithMix", 0x50A14DFC, since = 150) { __sceSasCoreWithMix(it) }
		registerFunctionRaw("__sceSasSetSL", 0x5F9529F6, since = 150) { __sceSasSetSL(it) }
		registerFunctionRaw("__sceSasGetEndFlag", 0x68A46B95, since = 150) { __sceSasGetEndFlag(it) }
		registerFunctionRaw("__sceSasGetEnvelopeHeight", 0x74AE582A, since = 150) { __sceSasGetEnvelopeHeight(it) }
		registerFunctionRaw("__sceSasSetKeyOn", 0x76F01ACA, since = 150) { __sceSasSetKeyOn(it) }
		registerFunctionRaw("__sceSasSetPause", 0x787D04D5, since = 150) { __sceSasSetPause(it) }
		registerFunctionRaw("__sceSasSetVoice", 0x99944089, since = 150) { __sceSasSetVoice(it) }
		registerFunctionRaw("__sceSasSetADSRmode", 0x9EC3676A, since = 150) { __sceSasSetADSRmode(it) }
		registerFunctionRaw("__sceSasSetKeyOff", 0xA0CF2FA4, since = 150) { __sceSasSetKeyOff(it) }
		registerFunctionRaw("__sceSasSetTrianglarWave", 0xA232CBE6, since = 150) { __sceSasSetTrianglarWave(it) }
		registerFunctionRaw("__sceSasCore", 0xA3589D81, since = 150) { __sceSasCore(it) }
		registerFunctionRaw("__sceSasSetPitch", 0xAD84D37F, since = 150) { __sceSasSetPitch(it) }
		registerFunctionRaw("__sceSasSetNoise", 0xB7660A23, since = 150) { __sceSasSetNoise(it) }
		registerFunctionRaw("__sceSasGetGrain", 0xBD11B7C2, since = 150) { __sceSasGetGrain(it) }
		registerFunctionRaw("__sceSasSetSimpleADSR", 0xCBCD4F79, since = 150) { __sceSasSetSimpleADSR(it) }
		registerFunctionRaw("__sceSasSetGrain", 0xD1E0A01E, since = 150) { __sceSasSetGrain(it) }
		registerFunctionRaw("__sceSasRevEVOL", 0xD5A229C9, since = 150) { __sceSasRevEVOL(it) }
		registerFunctionRaw("__sceSasSetSteepWave", 0xD5EBBBCD, since = 150) { __sceSasSetSteepWave(it) }
		registerFunctionRaw("__sceSasGetOutputmode", 0xE175EF66, since = 150) { __sceSasGetOutputmode(it) }
		registerFunctionRaw("__sceSasSetVoicePCM", 0xE1CD9561, since = 150) { __sceSasSetVoicePCM(it) }
		registerFunctionRaw("__sceSasSetOutputmode", 0xE855BF76, since = 150) { __sceSasSetOutputmode(it) }
		registerFunctionRaw("__sceSasRevVON", 0xF983B186, since = 150) { __sceSasRevVON(it) }
	}
}
