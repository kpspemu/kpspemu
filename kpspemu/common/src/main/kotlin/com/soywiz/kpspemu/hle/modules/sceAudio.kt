package com.soywiz.kpspemu.hle.modules


import com.soywiz.korio.async.sleep
import com.soywiz.korio.coroutine.getCoroutineContext
import com.soywiz.kpspemu.Emulator
import com.soywiz.kpspemu.cpu.CpuState
import com.soywiz.kpspemu.hle.SceModule
import com.soywiz.kpspemu.mem.Ptr


class sceAudio(emulator: Emulator) : SceModule(emulator, "sceAudio", 0x40010011, "popsman.prx", "scePops_Manager") {
	object AudioFormat {
		val Stereo = 0x00
		val Mono = 0x10
	}

	fun sceAudioChReserve(channelId: Int, sampleCount: Int, audioFormat: Int): Int {
		logger.info("WIP: sceAudioChReserve")
		return 0
	}

	suspend fun sceAudioOutputPannedBlocking(channelId: Int, leftVolume: Int, rightVolume: Int, ptr: Ptr): Int {
		logger.info("WIP: sceAudioOutputPannedBlocking")
		getCoroutineContext().sleep(10)
		return 0
	}

	fun sceAudioChReserve(cpu: CpuState): Unit = UNIMPLEMENTED(0x5EC81C55)

	fun sceAudioOutput2Reserve(cpu: CpuState): Unit = UNIMPLEMENTED(0x01562BA3)
	fun sceAudioInputBlocking(cpu: CpuState): Unit = UNIMPLEMENTED(0x086E5895)
	fun sceAudioOutputBlocking(cpu: CpuState): Unit = UNIMPLEMENTED(0x136CAF51)
	fun sceAudioOutput2OutputBlocking(cpu: CpuState): Unit = UNIMPLEMENTED(0x2D53F36E)
	fun sceAudioSRCChReserve(cpu: CpuState): Unit = UNIMPLEMENTED(0x38553111)
	fun sceAudioOneshotOutput(cpu: CpuState): Unit = UNIMPLEMENTED(0x41EFADE7)
	fun sceAudioOutput2Release(cpu: CpuState): Unit = UNIMPLEMENTED(0x43196845)
	fun sceAudioSRCChRelease(cpu: CpuState): Unit = UNIMPLEMENTED(0x5C37C0AE)
	fun sceAudioOutput2ChangeLength(cpu: CpuState): Unit = UNIMPLEMENTED(0x63F2889C)
	fun sceAudioOutput2GetRestSample(cpu: CpuState): Unit = UNIMPLEMENTED(0x647CEF33)
	fun sceAudioInput(cpu: CpuState): Unit = UNIMPLEMENTED(0x6D4BEC68)
	fun sceAudioChRelease(cpu: CpuState): Unit = UNIMPLEMENTED(0x6FC46853)
	fun sceAudioInputInit(cpu: CpuState): Unit = UNIMPLEMENTED(0x7DE61688)
	fun sceAudioWaitInputEnd(cpu: CpuState): Unit = UNIMPLEMENTED(0x87B2E651)
	fun sceAudioOutput(cpu: CpuState): Unit = UNIMPLEMENTED(0x8C1009B2)
	fun sceAudioChangeChannelConfig(cpu: CpuState): Unit = UNIMPLEMENTED(0x95FD0C2D)
	fun sceAudioPollInputEnd(cpu: CpuState): Unit = UNIMPLEMENTED(0xA633048E)
	fun sceAudioGetInputLength(cpu: CpuState): Unit = UNIMPLEMENTED(0xA708C6A6)
	fun sceAudioGetChannelRestLength(cpu: CpuState): Unit = UNIMPLEMENTED(0xB011922F)
	fun sceAudioChangeChannelVolume(cpu: CpuState): Unit = UNIMPLEMENTED(0xB7E1D8E7)
	fun sceAudioSetChannelDataLen(cpu: CpuState): Unit = UNIMPLEMENTED(0xCB2E439E)
	fun sceAudioSRCOutputBlocking(cpu: CpuState): Unit = UNIMPLEMENTED(0xE0727056)
	fun sceAudioOutputPanned(cpu: CpuState): Unit = UNIMPLEMENTED(0xE2D56B2D)
	fun sceAudioInputInitEx(cpu: CpuState): Unit = UNIMPLEMENTED(0xE926D3FB)
	fun sceAudioGetChannelRestLen(cpu: CpuState): Unit = UNIMPLEMENTED(0xE9D97901)


	override fun registerModule() {
		registerFunctionInt("sceAudioChReserve", 0x5EC81C55, since = 150) { sceAudioChReserve(int, int, int) }
		registerFunctionSuspendInt("sceAudioOutputPannedBlocking", 0x13F592BC, since = 150) { sceAudioOutputPannedBlocking(int, int, int, ptr) }

		registerFunctionRaw("sceAudioOutput2Reserve", 0x01562BA3, since = 150) { sceAudioOutput2Reserve(it) }
		registerFunctionRaw("sceAudioInputBlocking", 0x086E5895, since = 150) { sceAudioInputBlocking(it) }
		registerFunctionRaw("sceAudioOutputBlocking", 0x136CAF51, since = 150) { sceAudioOutputBlocking(it) }
		registerFunctionRaw("sceAudioOutput2OutputBlocking", 0x2D53F36E, since = 150) { sceAudioOutput2OutputBlocking(it) }
		registerFunctionRaw("sceAudioSRCChReserve", 0x38553111, since = 150) { sceAudioSRCChReserve(it) }
		registerFunctionRaw("sceAudioOneshotOutput", 0x41EFADE7, since = 150) { sceAudioOneshotOutput(it) }
		registerFunctionRaw("sceAudioOutput2Release", 0x43196845, since = 150) { sceAudioOutput2Release(it) }
		registerFunctionRaw("sceAudioSRCChRelease", 0x5C37C0AE, since = 150) { sceAudioSRCChRelease(it) }
		registerFunctionRaw("sceAudioOutput2ChangeLength", 0x63F2889C, since = 150) { sceAudioOutput2ChangeLength(it) }
		registerFunctionRaw("sceAudioOutput2GetRestSample", 0x647CEF33, since = 150) { sceAudioOutput2GetRestSample(it) }
		registerFunctionRaw("sceAudioInput", 0x6D4BEC68, since = 150) { sceAudioInput(it) }
		registerFunctionRaw("sceAudioChRelease", 0x6FC46853, since = 150) { sceAudioChRelease(it) }
		registerFunctionRaw("sceAudioInputInit", 0x7DE61688, since = 150) { sceAudioInputInit(it) }
		registerFunctionRaw("sceAudioWaitInputEnd", 0x87B2E651, since = 150) { sceAudioWaitInputEnd(it) }
		registerFunctionRaw("sceAudioOutput", 0x8C1009B2, since = 150) { sceAudioOutput(it) }
		registerFunctionRaw("sceAudioChangeChannelConfig", 0x95FD0C2D, since = 150) { sceAudioChangeChannelConfig(it) }
		registerFunctionRaw("sceAudioPollInputEnd", 0xA633048E, since = 150) { sceAudioPollInputEnd(it) }
		registerFunctionRaw("sceAudioGetInputLength", 0xA708C6A6, since = 150) { sceAudioGetInputLength(it) }
		registerFunctionRaw("sceAudioGetChannelRestLength", 0xB011922F, since = 150) { sceAudioGetChannelRestLength(it) }
		registerFunctionRaw("sceAudioChangeChannelVolume", 0xB7E1D8E7, since = 150) { sceAudioChangeChannelVolume(it) }
		registerFunctionRaw("sceAudioSetChannelDataLen", 0xCB2E439E, since = 150) { sceAudioSetChannelDataLen(it) }
		registerFunctionRaw("sceAudioSRCOutputBlocking", 0xE0727056, since = 150) { sceAudioSRCOutputBlocking(it) }
		registerFunctionRaw("sceAudioOutputPanned", 0xE2D56B2D, since = 150) { sceAudioOutputPanned(it) }
		registerFunctionRaw("sceAudioInputInitEx", 0xE926D3FB, since = 150) { sceAudioInputInitEx(it) }
		registerFunctionRaw("sceAudioGetChannelRestLen", 0xE9D97901, since = 150) { sceAudioGetChannelRestLen(it) }
	}
}
