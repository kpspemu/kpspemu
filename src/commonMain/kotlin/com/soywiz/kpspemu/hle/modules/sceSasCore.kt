package com.soywiz.kpspemu.hle.modules

import com.soywiz.kmem.*
import com.soywiz.korio.stream.*
import com.soywiz.korio.util.*
import com.soywiz.kpspemu.*
import com.soywiz.kpspemu.hle.*
import com.soywiz.kpspemu.hle.error.*
import com.soywiz.kpspemu.mem.*
import com.soywiz.kpspemu.util.*
import com.soywiz.krypto.encoding.*

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
        const val PSP_SAS_ADSR_CURVE_MODE_LINEAR_INCREASE = 0
        const val PSP_SAS_ADSR_CURVE_MODE_LINEAR_DECREASE = 1
        const val PSP_SAS_ADSR_CURVE_MODE_LINEAR_BENT = 2
        const val PSP_SAS_ADSR_CURVE_MODE_EXPONENT_DECREASE = 3
        const val PSP_SAS_ADSR_CURVE_MODE_EXPONENT_INCREASE = 4
        const val PSP_SAS_ADSR_CURVE_MODE_DIRECT = 5
        const val PSP_SAS_ADSR_ATTACK = 1
        const val PSP_SAS_ADSR_DECAY = 2
        const val PSP_SAS_ADSR_SUSTAIN = 4
        const val PSP_SAS_ADSR_RELEASE = 8
        const val PSP_SAS_OUTPUTMODE_STEREO = 0
        const val PSP_SAS_OUTPUTMODE_MONO = 1
        const val PSP_SAS_EFFECT_TYPE_OFF = -1
        const val PSP_SAS_EFFECT_TYPE_ROOM = 0
        const val PSP_SAS_EFFECT_TYPE_UNK1 = 1
        const val PSP_SAS_EFFECT_TYPE_UNK2 = 2
        const val PSP_SAS_EFFECT_TYPE_UNK3 = 3
        const val PSP_SAS_EFFECT_TYPE_HALL = 4
        const val PSP_SAS_EFFECT_TYPE_SPACE = 5
        const val PSP_SAS_EFFECT_TYPE_ECHO = 6
        const val PSP_SAS_EFFECT_TYPE_DELAY = 7
        const val PSP_SAS_EFFECT_TYPE_PIPE = 8

    }

    private val core = SasCore()

    fun __sceSasCoreWithMix(sas: Int, sasInOut: Int, leftVolume: Int, rightVolume: Int): Int {
        logger.trace { "__sceSasCoreWithMix" }
        return 0
    }

    fun __sceSasCore(sas: Int, sasInOut: Int): Int = __sceSasCoreWithMix(sas, sasInOut, 0x1000, 0x1000)

    fun __sceSasInit(sasCorePointer: Int, grainSamples: Int, maxVoices: Int, outputMode: Int, sampleRate: Int): Int {
        if (sampleRate != 44100) sceKernelException(SceKernelErrors.ERROR_SAS_INVALID_SAMPLE_RATE)
        if (maxVoices < 1 || maxVoices > sceSasCore.PSP_SAS_VOICES_MAX) sceKernelException(SceKernelErrors.ERROR_SAS_INVALID_MAX_VOICES)
        if (outputMode != OutputMode.STEREO.id && outputMode != OutputMode.MULTICHANNEL.id) sceKernelException(
            SceKernelErrors.ERROR_SAS_INVALID_OUTPUT_MODE
        )

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

    //private fun hasSasCoreVoice(sas: Int, voiceId: Int) = this.core.voices.getOrNull(voiceId) != null
    private fun getVoice(sas: Int, voiceId: Int): Voice =
        this.core.voices.getOrNull(voiceId) ?: sceKernelException(SceKernelErrors.ERROR_SAS_INVALID_VOICE)

    //@nativeFunction(0x019B25EB, 150, 'uint', 'int/int/int/int/int/int/int', { originalName: "__sceSasSetADSR" })
    fun __sceSasSetADSR(
        sasCorePointer: Int,
        voiceId: Int,
        flags: Int,
        attackRate: Int,
        decayRate: Int,
        sustainRate: Int,
        releaseRate: Int
    ): Int {
        val voice = this.getVoice(sasCorePointer, voiceId)

        if (flags hasFlag AdsrFlags.HasAttack) voice.envelope.attackRate = attackRate
        if (flags hasFlag AdsrFlags.HasDecay) voice.envelope.decayRate = decayRate
        if (flags hasFlag AdsrFlags.HasSustain) voice.envelope.sustainRate = sustainRate
        if (flags hasFlag AdsrFlags.HasRelease) voice.envelope.releaseRate = releaseRate
        logger.warn { "__sceSasSetADSR" }
        return 0
    }

    fun __sceSasGetEndFlag(sas: Int): Int {
        var endFlag = 0
        for ((i, voice) in core.voices.withIndex()) {
            if (voice.ended) endFlag = endFlag or (1 shl i)
        }
        return endFlag
    }

    fun __sceSasSetVolume(
        sas: Int,
        voice: Int,
        leftVolume: Int,
        rightVolume: Int,
        effectLeftVolume: Int,
        effectRightVolume: Int
    ): Int {
        val v = getVoice(sas, voice)
        // 0 - 0x1000
        v.leftVolume = leftVolume * 8
        v.rightVolume = rightVolume * 8
        v.effectLeftVolume = effectLeftVolume * 8
        v.effectRightVolume = effectRightVolume * 8
        return 0
    }

    fun __sceSasSetKeyOnOff(sas: Int, voice: Int, on: Boolean): Int {
        val v = getVoice(sas, voice)
        if (v.paused == on) return SceKernelErrors.ERROR_SAS_VOICE_PAUSED
        v.setOn(on)
        return 0
    }

    fun __sceSasRevType(sas: Int, type: Int): Int = 0.apply { core.waveformEffectType = WaveformEffectType(type) }
    fun __sceSasSetPitch(sas: Int, voice: Int, pitch: Int): Int = 0.apply { getVoice(sas, voice).pitch = pitch }
    fun __sceSasSetKeyOn(sas: Int, voice: Int): Int = __sceSasSetKeyOnOff(sas, voice, on = true)
    fun __sceSasSetKeyOff(sas: Int, voice: Int): Int = __sceSasSetKeyOnOff(sas, voice, on = false)
    fun __sceSasSetSL(sas: Int, voice: Int, level: Int): Int =
        0.apply { getVoice(sas, voice).envelope.sustainLevel = level }

    fun __sceSasSetNoise(sas: Int, voice: Int, freq: Int): Int = 0.apply { getVoice(sas, voice).noise = freq }
    fun __sceSasSetGrain(sas: Int, grain: Int): Int = 0.apply { core.grainSamples = grain }
    fun __sceSasSetPause(sas: Int, voiceBits: Int, setPause: Boolean): Int {
        for (n in 0 until 32) {
            if (voiceBits hasFlag (1 shl n)) core.voices[n].paused = true
        }
        return 0
    }

    fun __sceSasSetVoice(sas: Int, voice: Int, vagAddr: Ptr, size: Int, loopCount: Int): Int {
        logger.warn { "__sceSasSetVoice: ${vagAddr.addr.hex}, size=$size, loopmode=$loopCount" }
        val v = getVoice(sas, voice)
        v.setAdpcm(vagAddr.openSync().readSlice(size.toLong()), loopCount)
        return 0
    }

    fun __sceSasSetVoicePCM(sas: Int, voice: Int, pcmAddr: Ptr, size: Int, loopCount: Int): Int {
        logger.warn { "__sceSasSetVoicePCM: ${pcmAddr.addr.hex}, size=$size, loopmode=$loopCount" }
        val v = getVoice(sas, voice)
        v.setPCM(pcmAddr.openSync().readSlice(size.toLong()), loopCount)
        return 0
    }

    fun __sceSasSetSteepWave(sas: Int, voice: Int, steepWave: Int): Int =
        0.apply { getVoice(sas, voice).steepWave = steepWave }

    fun __sceSasSetADSRmode(
        sas: Int,
        voice: Int,
        flag: Int,
        attackType: Int,
        decayType: Int,
        sustainType: Int,
        releaseType: Int
    ): Int {
        val envelope = getVoice(sas, voice).envelope
        if (flag and 0x1 != 0) envelope.attackCurveType = attackType
        if (flag and 0x2 != 0) envelope.decayCurveType = decayType
        if (flag and 0x4 != 0) envelope.sustainCurveType = sustainType
        if (flag and 0x8 != 0) envelope.releaseCurveType = releaseType
        return 0
    }

    fun __sceSasSetOutputmode(sas: Int, outputMode: Int): Int = 0.apply { core.outputMode = OutputMode(outputMode) }
    fun __sceSasGetGrain(sas: Int): Int = core.grainSamples

    fun __sceSasRevParam(sas: Int, delay: Int, feedback: Int): Int = 0.apply {
        core.waveformEffectDelay = delay
        core.waveformEffectFeedback = delay
    }

    fun __sceSasRevEVOL(sas: Int, lvol: Int, rvol: Int): Int = 0.apply {
        core.waveformEffectLeftVol = lvol
        core.waveformEffectRightVol = rvol
    }

    fun __sceSasRevVON(sas: Int, dry: Int, wet: Int): Int = 0.apply {
        core.waveformEffectIsDry = dry > 0
        core.waveformEffectIsWet = wet > 0
    }

    fun __sceSasGetOutputmode(sas: Int): Int = core.outputMode.id
    fun __sceSasSetTrianglarWave(sas: Int, voice: Int, triangularWave: Int): Int = 0.apply {
        getVoice(sas, voice).triangularWave = triangularWave
    }

    fun __sceSasSetSimpleADSR(sas: Int, voice: Int, adsrEnv1: Int, adsrEnv2: Int): Int {
        val v = getVoice(sas, voice)
        val envelope = v.envelope
        val e1 = adsrEnv1 and 0xFFFF
        val d2 = adsrEnv2 and 0xFFFF
        envelope.sustainLevel = getSimpleSustainLevel(e1)
        envelope.decayRate = getSimpleDecayRate(e1)
        envelope.decayCurveType = PSP_SAS_ADSR_CURVE_MODE_EXPONENT_DECREASE
        envelope.attackRate = getSimpleAttackRate(e1)
        envelope.attackCurveType = getSimpleAttackCurveType(e1)

        envelope.releaseRate = getSimpleReleaseRate(d2)
        envelope.releaseCurveType = getSimpleReleaseCurveType(d2)
        envelope.sustainRate = getSimpleSustainRate(d2)
        envelope.sustainCurveType = getSimpleSustainCurveType(d2)
        return 0
    }

    fun __sceSasGetPauseFlag(sas: Int): Int {
        var pauseFlag = 0
        for (i in core.voices.indices) {
            if (core.voices[i].paused) pauseFlag = pauseFlag or (1 shl i)
        }
        return pauseFlag
    }

    fun __sceSasGetEnvelopeHeight(sas: Int, voice: Int): Int = core.voices[voice].envelope.height

    fun __sceSasGetAllEnvelopeHeights(sas: Int, heights: Ptr32): Int = 0.apply {
        for (n in 0 until 32) heights[n] = core.voices[n].height
    }


    override fun registerModule() {
        registerFunctionInt("__sceSasInit", 0x42778A9F, since = 150) { __sceSasInit(int, int, int, int, int) }
        registerFunctionInt("__sceSasSetADSR", 0x019B25EB, since = 150) {
            __sceSasSetADSR(
                int,
                int,
                int,
                int,
                int,
                int,
                int
            )
        }
        registerFunctionInt("__sceSasGetEndFlag", 0x68A46B95, since = 150) { __sceSasGetEndFlag(int) }
        registerFunctionInt("__sceSasRevType", 0x33D4AB37, since = 150) { __sceSasRevType(int, int) }
        registerFunctionInt("__sceSasSetVolume", 0x440CA7D8, since = 150) {
            __sceSasSetVolume(
                int,
                int,
                int,
                int,
                int,
                int
            )
        }
        registerFunctionInt("__sceSasCoreWithMix", 0x50A14DFC, since = 150) { __sceSasCoreWithMix(int, int, int, int) }
        registerFunctionInt("__sceSasCore", 0xA3589D81, since = 150) { __sceSasCore(int, int) }
        registerFunctionInt("__sceSasSetPitch", 0xAD84D37F, since = 150) { __sceSasSetPitch(int, int, int) }
        registerFunctionInt("__sceSasSetKeyOn", 0x76F01ACA, since = 150) { __sceSasSetKeyOn(int, int) }
        registerFunctionInt("__sceSasSetKeyOff", 0xA0CF2FA4, since = 150) { __sceSasSetKeyOff(int, int) }
        registerFunctionInt("__sceSasSetSL", 0x5F9529F6, since = 150) { __sceSasSetSL(int, int, int) }

        registerFunctionInt("__sceSasGetAllEnvelopeHeights", 0x07F58C24, since = 150) {
            __sceSasGetAllEnvelopeHeights(
                int,
                ptr32
            )
        }
        registerFunctionInt("__sceSasRevParam", 0x267A6DD2, since = 150) { __sceSasRevParam(int, int, int) }
        registerFunctionInt("__sceSasGetPauseFlag", 0x2C8E6AB3, since = 150) { __sceSasGetPauseFlag(int) }
        registerFunctionInt("__sceSasGetEnvelopeHeight", 0x74AE582A, since = 150) {
            __sceSasGetEnvelopeHeight(
                int,
                int
            )
        }
        registerFunctionInt("__sceSasSetPause", 0x787D04D5, since = 150) { __sceSasSetPause(int, int, bool) }
        registerFunctionInt("__sceSasSetVoice", 0x99944089, since = 150) { __sceSasSetVoice(int, int, ptr, int, int) }
        registerFunctionInt("__sceSasSetADSRmode", 0x9EC3676A, since = 150) {
            __sceSasSetADSRmode(
                int,
                int,
                int,
                int,
                int,
                int,
                int
            )
        }
        registerFunctionInt("__sceSasSetTrianglarWave", 0xA232CBE6, since = 150) {
            __sceSasSetTrianglarWave(
                int,
                int,
                int
            )
        }
        registerFunctionInt("__sceSasSetNoise", 0xB7660A23, since = 150) { __sceSasSetNoise(int, int, int) }
        registerFunctionInt("__sceSasGetGrain", 0xBD11B7C2, since = 150) { __sceSasGetGrain(int) }
        registerFunctionInt("__sceSasSetSimpleADSR", 0xCBCD4F79, since = 150) {
            __sceSasSetSimpleADSR(
                int,
                int,
                int,
                int
            )
        }
        registerFunctionInt("__sceSasSetGrain", 0xD1E0A01E, since = 150) { __sceSasSetGrain(int, int) }
        registerFunctionInt("__sceSasRevEVOL", 0xD5A229C9, since = 150) { __sceSasRevEVOL(int, int, int) }
        registerFunctionInt("__sceSasSetSteepWave", 0xD5EBBBCD, since = 150) { __sceSasSetSteepWave(int, int, int) }
        registerFunctionInt("__sceSasGetOutputmode", 0xE175EF66, since = 150) { __sceSasGetOutputmode(int) }
        registerFunctionInt("__sceSasSetVoicePCM", 0xE1CD9561, since = 150) {
            __sceSasSetVoicePCM(
                int,
                int,
                ptr,
                int,
                int
            )
        }
        registerFunctionInt("__sceSasSetOutputmode", 0xE855BF76, since = 150) { __sceSasSetOutputmode(int, int) }
        registerFunctionInt("__sceSasRevVON", 0xF983B186, since = 150) { __sceSasRevVON(int, int, int) }
    }

    enum class WaveformEffectType(override val id: Int) : IdEnum {
        OFF(-1), ROOM(0), UNK1(1), UNK2(2), UNK3(3),
        HALL(4), SPACE(5), ECHO(6), DELAY(7), PIPE(8);

        companion object : SmallCompanion2<WaveformEffectType>(values())
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
        var voices: Array<Voice> = Array(32) { Voice(it) }
        var bufferTempArray = arrayListOf<List<Sample>>()
        var waveformEffectDelay: Int = 0
        var waveformEffectFeedback: Int = 0
        var waveformEffectLeftVol: Int = 0
        var waveformEffectRightVol: Int = 0
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
        var noise: Int = 0
        var steepWave: Int = 0

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
            //this.source = TODO("VagSoundSource")
            this.source = null
            this.source?.reset()
        }

        fun setPCM(stream: SyncStream, loopCount: Int) {
            this.source = PcmSoundSource(stream, loopCount)
            this.source?.reset()
        }

        var triangularWave: Int = 0
        val height: Int = 0

    }

    class PcmSoundSource(val stream: SyncStream, val loopCount: Int) : SoundSource {
        val sample = Sample(0.0, 0.0)
        override fun reset() = Unit
        override val hasMore: Boolean get() = false // @TODO:
        override fun getNextSample(): Sample {
            // @TODO:
            return sample
        }
    }


    class Envelope {
        var attackRate = 0
        var decayRate = 0
        var sustainRate = 0
        var releaseRate = 0
        var height = 0
        var sustainLevel = 0
        var attackCurveType: Int = 0
        var decayCurveType: Int = 0
        var sustainCurveType: Int = 0
        var releaseCurveType: Int = 0
    }

    enum class OutputMode(override val id: Int) : IdEnum {
        STEREO(0), MULTICHANNEL(1);

        companion object : IdEnum.SmallCompanion<OutputMode>(values())
    }

    enum class AdsrCurveMode(override val id: Int) : IdEnum {
        LINEAR_INCREASE(0), LINEAR_DECREASE(1), LINEAR_BENT(2),
        EXPONENT_REV(3), EXPONENT(4), DIRECT(5);

        companion object : IdEnum.SmallCompanion<AdsrCurveMode>(values())
    }

    object AdsrFlags {
        const val HasAttack = (1 shl 0)
        const val HasDecay = (1 shl 1)
        const val HasSustain = (1 shl 2)
        const val HasRelease = (1 shl 3)
    }

    fun getSimpleRate(n: Int): Int {
        var n = n
        n = n and 0x7F
        if (n == 0x7F) return 0
        val rate = (7 - (n and 0x3) shl 26).ushr(n shr 2)
        return if (rate == 0) 1 else rate
    }

    fun getSimpleSustainLevel(bitfield1: Int): Int = (bitfield1 and 0x000F) + 1 shl 26

    fun getSimpleDecayRate(bitfield1: Int): Int {
        val bitShift = bitfield1 shr 4 and 0x000F
        return if (bitShift == 0) PSP_SAS_ENVELOPE_FREQ_MAX else -0x80000000.toInt().ushr(bitShift)
    }

    fun getSimpleExponentRate(n: Int): Int {
        var n = n
        n = n and 0x7F
        if (n == 0x7F) return 0
        val rate = (7 - (n and 0x3) shl 24).ushr(n shr 2)
        return if (rate == 0) 1 else rate
    }

    fun getSimpleAttackRate(bitfield1: Int): Int = getSimpleRate(bitfield1 shr 8)
    fun getSimpleAttackCurveType(bitfield1: Int): Int =
        if (bitfield1 and 0x8000 == 0) PSP_SAS_ADSR_CURVE_MODE_LINEAR_INCREASE else PSP_SAS_ADSR_CURVE_MODE_LINEAR_BENT

    fun getSimpleReleaseCurveType(bitfield2: Int): Int =
        if (bitfield2 and 0x0020 == 0) PSP_SAS_ADSR_CURVE_MODE_LINEAR_DECREASE else PSP_SAS_ADSR_CURVE_MODE_EXPONENT_DECREASE

    fun getSimpleReleaseRate(bitfield2: Int): Int {
        val n = bitfield2 and 0x001F
        if (n == 31) {
            return 0
        }
        if (getSimpleReleaseCurveType(bitfield2) == PSP_SAS_ADSR_CURVE_MODE_LINEAR_DECREASE) {
            if (n == 30) {
                return 0x40000000
            } else if (n == 29) {
                return 1
            }
            return 0x10000000 shr n
        }
        return if (n == 0) {
            PSP_SAS_ENVELOPE_FREQ_MAX
        } else -0x80000000.toInt().ushr(n)
    }

    fun getSimpleSustainCurveType(bitfield2: Int): Int {
        when (bitfield2 shr 13) {
            0 -> return PSP_SAS_ADSR_CURVE_MODE_LINEAR_INCREASE
            2 -> return PSP_SAS_ADSR_CURVE_MODE_LINEAR_DECREASE
            4 -> return PSP_SAS_ADSR_CURVE_MODE_LINEAR_BENT
            6 -> return PSP_SAS_ADSR_CURVE_MODE_EXPONENT_DECREASE
        }

        sceKernelException(SceKernelErrors.ERROR_SAS_INVALID_ADSR_CURVE_MODE)
    }

    fun getSimpleSustainRate(bitfield2: Int): Int {
        return if (getSimpleSustainCurveType(bitfield2) == PSP_SAS_ADSR_CURVE_MODE_EXPONENT_DECREASE) {
            getSimpleExponentRate(bitfield2 shr 6)
        } else getSimpleRate(bitfield2 shr 6)
    }
}
