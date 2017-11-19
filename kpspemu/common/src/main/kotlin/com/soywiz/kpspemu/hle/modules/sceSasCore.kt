package com.soywiz.kpspemu.hle.modules

import com.soywiz.korio.stream.SyncStream
import com.soywiz.korio.util.IdEnum
import com.soywiz.kpspemu.Emulator
import com.soywiz.kpspemu.cpu.CpuState
import com.soywiz.kpspemu.hle.SceModule
import com.soywiz.kpspemu.hle.error.SceKernelErrors
import com.soywiz.kpspemu.hle.error.sceKernelException

@Suppress("UNUSED_PARAMETER")
class sceSasCore(emulator: Emulator) : SceModule(emulator, "sceSasCore", 0x40010011, "sc_sascore.prx", "sceSAScore") {
	companion object {
		const val PSP_SAS_VOL_MAX = 0x1000
		const val PSP_SAS_PITCH_MIN = 0x1
		const val PSP_SAS_PITCH_BASE = 0x1000
		const val PSP_SAS_PITCH_MAX = 0x4000
		const val PSP_SAS_VOICES_MAX = 32
		const val PSP_SAS_GRAIN_SAMPLES = 256
		const val PSP_SAS_LOOP_MODE_OFF = 0
		const val PSP_SAS_LOOP_MODE_ON = 1
		const val PSP_SAS_NOISE_FREQ_MAX = 0x3F
		const val PSP_SAS_ENVELOPE_HEIGHT_MAX = 0x40000000
		const val PSP_SAS_ENVELOPE_FREQ_MAX = 0x7FFFFFFF
		const val PSP_SAS_ADSR_ATTACK = 1
		const val PSP_SAS_ADSR_DECAY = 2
		const val PSP_SAS_ADSR_SUSTAIN = 4
		const val PSP_SAS_ADSR_RELEASE = 8
	}

	private val core = SasCore()

	fun __sceSasInit(sasCorePointer: Int, grainSamples: Int, maxVoices: Int, outputMode: Int, sampleRate: Int): Int {
		if (sampleRate != 44100) sceKernelException(SceKernelErrors.ERROR_SAS_INVALID_SAMPLE_RATE)
		if (maxVoices < 1 || maxVoices > sceSasCore.PSP_SAS_VOICES_MAX) sceKernelException(SceKernelErrors.ERROR_SAS_INVALID_MAX_VOICES)
		if (outputMode != OutputMode.STEREO.id && outputMode != OutputMode.MULTICHANNEL.id) sceKernelException(SceKernelErrors.ERROR_SAS_INVALID_OUTPUT_MODE)

		//var SasCore = GetSasCore(SasCorePointer, CreateIfNotExists: true);
		this.core.grainSamples = grainSamples
		this.core.maxVoices = maxVoices
		this.core.outputMode = OutputMode(outputMode)
		this.core.sampleRate = sampleRate
		this.core.initialized = true

		//BufferTemp = new StereoIntSoundSample[SasCore.GrainSamples * 2];
		//BufferShort = new StereoShortSoundSample[SasCore.GrainSamples * 2];
		//MixBufferShort = new StereoShortSoundSample[SasCore.GrainSamples * 2];

		return 0
	}

	fun __sceSasSetADSR(cpu: CpuState): Unit = UNIMPLEMENTED(0x019B25EB)
	fun __sceSasGetAllEnvelopeHeights(cpu: CpuState): Unit = UNIMPLEMENTED(0x07F58C24)
	fun __sceSasRevParam(cpu: CpuState): Unit = UNIMPLEMENTED(0x267A6DD2)
	fun __sceSasGetPauseFlag(cpu: CpuState): Unit = UNIMPLEMENTED(0x2C8E6AB3)
	fun __sceSasRevType(cpu: CpuState): Unit = UNIMPLEMENTED(0x33D4AB37)
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
		registerFunctionInt("__sceSasInit", 0x42778A9F, since = 150) { __sceSasInit(int, int, int, int, int) }

		registerFunctionRaw("__sceSasSetADSR", 0x019B25EB, since = 150) { __sceSasSetADSR(it) }
		registerFunctionRaw("__sceSasGetAllEnvelopeHeights", 0x07F58C24, since = 150) { __sceSasGetAllEnvelopeHeights(it) }
		registerFunctionRaw("__sceSasRevParam", 0x267A6DD2, since = 150) { __sceSasRevParam(it) }
		registerFunctionRaw("__sceSasGetPauseFlag", 0x2C8E6AB3, since = 150) { __sceSasGetPauseFlag(it) }
		registerFunctionRaw("__sceSasRevType", 0x33D4AB37, since = 150) { __sceSasRevType(it) }
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

	class SasCore {
		var initialized = false
		var grainSamples = 0
		var maxVoices = 32
		var outputMode = OutputMode.STEREO
		var sampleRate = 44100
		var delay = 0
		var feedback = 0
		var endFlags = 0
		var waveformEffectType = WaveformEffectType.OFF
		var waveformEffectIsDry = false
		var waveformEffectIsWet = false
		var leftVolume = PSP_SAS_VOL_MAX
		var rightVolume = PSP_SAS_VOL_MAX
		var voices: ArrayList<Voice> = arrayListOf<Voice>()
		var bufferTempArray = arrayListOf<List<Sample>>()
	}

	interface SoundSource {
		val hasMore: Boolean
		fun reset(): Unit
		fun getNextSample(): Sample
	}

	class Sample(var left: Double, var right: Double) {
		fun set(left: Double, right: Double): Sample {
			this.left = left
			this.right = right
			return this
		}

		fun scale(leftScale: Double, rightScale: Double) {
			this.left *= leftScale
			this.right *= rightScale
		}

		fun addScaled(sample: Sample, leftScale: Double, rightScale: Double) {
			this.left += sample.left * leftScale
			this.right += sample.right * rightScale
		}

		fun GetNextSample() = Unit
	}

	class Voice(val index: Int) {
		var envelope = Envelope()
		var sustainLevel = 0
		var _on = false
		var _playing = false
		var paused = false
		var leftVolume = PSP_SAS_VOL_MAX
		var rightVolume = PSP_SAS_VOL_MAX
		var effectLeftVolume = PSP_SAS_VOL_MAX
		var effectRightVolume = PSP_SAS_VOL_MAX
		var pitch = PSP_SAS_PITCH_BASE
		var source: SoundSource? = null

		val onAndPlaying get() = this._on && this._playing

		fun setOn(set: Boolean) {
			this._on = set
			this.setPlaying(set)
		}

		fun setPlaying(set: Boolean) {
			this._playing = set

			// CHECK. Reset on change?
			this.source?.reset()
		}

		val ended get() = !this._playing
		fun unsetSource() {
			this.source = null
		}

		fun setAdpcm(stream: SyncStream, loopCount: Int) {
			//this.source = VagSoundSource(stream, loopCount)
			this.source = TODO("VagSoundSource")
			this.source?.reset()
		}

		fun setPCM(stream: SyncStream, loopCount: Int) {
			this.source = PcmSoundSource(stream, loopCount)
			this.source?.reset()
		}
	}

	class PcmSoundSource(val stream: SyncStream, val loopCount: Int) : SoundSource {
		override fun reset() = Unit
		override val hasMore: Boolean get() = false
		override fun getNextSample(): Sample = TODO()
	}


	class Envelope {
		var attackRate = 0
		var decayRate = 0
		var sustainRate = 0
		var releaseRate = 0
		var height = 0
	}

	enum class OutputMode(override val id: Int) : IdEnum {
		STEREO(0), MULTICHANNEL(1);

		companion object : IdEnum.SmallCompanion<OutputMode>(values())
	}

	enum class WaveformEffectType(override val id: Int) : IdEnum {
		OFF(-1), ROOM(0), UNK1(1), UNK2(2), UNK3(3),
		HALL(4), SPACE(5), ECHO(6), DELAY(7), PIPE(8);

		//companion object : IdEnum.SmallCompanion<WaveformEffectType>(values())
	}

	enum class AdsrCurveMode(override val id: Int) : IdEnum {
		LINEAR_INCREASE(0), LINEAR_DECREASE(1), LINEAR_BENT(2),
		EXPONENT_REV(3), EXPONENT(4), DIRECT(5);

		companion object : IdEnum.SmallCompanion<AdsrCurveMode>(values())
	}

	object AdsrFlags {
		val HasAttack = (1 shl 0)
		val HasDecay = (1 shl 1)
		val HasSustain = (1 shl 2)
		val HasRelease = (1 shl 3)
	}
}
