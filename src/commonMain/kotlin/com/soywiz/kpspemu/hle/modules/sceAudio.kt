package com.soywiz.kpspemu.hle.modules

import com.soywiz.kds.*
import com.soywiz.kmem.*
import com.soywiz.korau.sound.*
import com.soywiz.kpspemu.*
import com.soywiz.kpspemu.cpu.*
import com.soywiz.kpspemu.hle.*
import com.soywiz.kpspemu.mem.*
import kotlin.math.*

class MyAudioStream(hz: Int, nchannels: Int = 2) : AudioStream(hz, nchannels) {
    val deque = AudioSamplesDeque(nchannels)
    var closed = false

    override suspend fun read(out: AudioSamples, offset: Int, length: Int): Int {
        if (closed) return -1

        val result = deque.read(out, offset, length)

        if (result <= 0) {
            close()
        }
        //println("   AudioStream.read -> result=$result")
        return result
    }

    override suspend fun clone(): AudioStream {
        return MyAudioStream(rate, channels)
    }

    override fun close() {
        closed = true
    }

    fun stop() {
        close()
    }

    fun addSamples(data: ShortArray) {
        deque.writeInterleaved(data, 0)
    }
}

class sceAudio(emulator: Emulator) : SceModule(emulator, "sceAudio", 0x40010011, "popsman.prx", "scePops_Manager") {
    object AudioFormat {
        const val STEREO = 0x00
        const val MONO = 0x10
    }

    class AudioChannel(val id: Int) {
        var reserved: Boolean = false
        var streamInitialized = false
        val stream by lazy { streamInitialized = true; MyAudioStream(44100) }
        var audioFormat: Int = 0
        var line = ShortArray(0)
        var lineEx = ShortArray(0)
        val isStereo get() = audioFormat == AudioFormat.STEREO
        val shortsPerSamples get() = if (isStereo) 2 else 1
        var volumeLeft: Double = 1.0
        var volumeRight: Double = 1.0
        var sampleCount: Int = 0
            private set(value) {
                field = value
                line = ShortArray(sampleCount * shortsPerSamples)
                lineEx = ShortArray(sampleCount * 2)
            }

        fun reconfigure(audioFormat: Int = this.audioFormat, sampleCount: Int = this.sampleCount) {
            this.audioFormat = audioFormat
            this.sampleCount = sampleCount
        }

        fun ensureInitStream() = stream
        //var started = 0.0
        //var msBuffered = 0.0
        //val msPerLine: Int get() = ((sampleCount.toDouble() * 1000.0) / 44100.0).toInt()

        fun stop() {
            if (streamInitialized) {
                stream.stop()
            }
        }
    }

    val channels = (0 until 32).map { AudioChannel(it) }

    override fun stopModule() {
        for (channel in channels) {
            channel.stop()
        }
    }

    var lastId = 0
    val pool = Pool { AudioChannel(lastId++) }

    fun sceAudioChReserve(channelId: Int, sampleCount: Int, audioFormat: Int): Int {
        println("WIP: sceAudioChReserve: $channelId, $sampleCount, $audioFormat")
        val actualChannelId = if (channelId >= 0) channelId else channels.indexOfFirst { !it.reserved }
        val channel = channels[actualChannelId]
        logger.info { "WIP: sceAudioChReserve: $channelId, $sampleCount, $audioFormat" }
        channel.ensureInitStream()
        //channel.started = timeManager.getTimeInMillisecondsDouble()
        //channel.msBuffered = 0.0
        channel.reserved = true
        channel.reconfigure(audioFormat = audioFormat, sampleCount = sampleCount)
        println(" sceAudioChReserve ---> $actualChannelId")
        return actualChannelId
    }

    suspend fun _sceAudioOutputPannedBlocking(channelId: Int, leftVolume: Double, rightVolume: Double, ptr: Int): Int {
        //println("WIP: sceAudioOutputPannedBlocking: ChannelId($channelId), Volumes($leftVolume, $rightVolume), Ptr(${ptr.hex32})")
        logger.trace { "WIP: sceAudioOutputPannedBlocking: ChannelId($channelId), Volumes($leftVolume, $rightVolume), Ptr($ptr)" }
        val channel = channels[channelId]
        mem.read(ptr, channel.line, 0, channel.line.size)
        if (channel.isStereo) {
            arraycopy(channel.line, 0, channel.lineEx, 0, min(channel.line.size, channel.lineEx.size))
        } else {
            var m = 0
            for (n in 0 until channel.sampleCount) {
                val sh = channel.line[n]
                channel.lineEx[m++] = sh
                channel.lineEx[m++] = sh
            }
        }

        channel.stream.addSamples(channel.lineEx)
        return 0
    }

    suspend fun sceAudioOutputPannedBlocking(channelId: Int, leftVolume: Int, rightVolume: Int, ptr: Int): Int {
        val channel = channels[channelId]
        //println("sceAudioOutputPannedBlocking")
        // @TODO: Verify we multiply by default channel volumes
        return _sceAudioOutputPannedBlocking(
            channelId,
            (leftVolume.shortVolumeToDouble() * channel.volumeLeft),
            (rightVolume.shortVolumeToDouble() * channel.volumeRight),
            ptr
        )
    }

    suspend fun sceAudioOutputBlocking(channelId: Int, volume: Int, ptr: Int): Int {
        val channel = channels[channelId]
        return _sceAudioOutputPannedBlocking(
            channelId,
            volume.shortVolumeToDouble() * channel.volumeLeft,
            volume.shortVolumeToDouble() * channel.volumeRight,
            ptr
        )
    }

    private fun Int.shortVolumeToDouble() = this.toDouble() / 32767.0

    fun sceAudioChangeChannelVolume(channelId: Int, volumeLeft: Int, volumeRight: Int): Int {
        logger.info() { "sceAudioChangeChannelVolume not implemented! $volumeLeft, $volumeRight" }
        val channel = channels[channelId]
        channel.volumeLeft = volumeLeft.shortVolumeToDouble()
        channel.volumeRight = volumeRight.shortVolumeToDouble()
        return 0
    }

    fun sceAudioSetChannelDataLen(channelId: Int, sampleCount: Int): Int {
        val channel = channels[channelId]
        channel.reconfigure(sampleCount = sampleCount)
        return 0
    }

    fun sceAudioChangeChannelConfig(channelId: Int, format: Int): Int {
        val channel = channels[channelId]
        channel.reconfigure(audioFormat = format)
        return 0
    }

    fun sceAudioOutput2Reserve(cpu: CpuState): Unit = UNIMPLEMENTED(0x01562BA3)
    fun sceAudioInputBlocking(cpu: CpuState): Unit = UNIMPLEMENTED(0x086E5895)
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
    fun sceAudioPollInputEnd(cpu: CpuState): Unit = UNIMPLEMENTED(0xA633048E)
    fun sceAudioGetInputLength(cpu: CpuState): Unit = UNIMPLEMENTED(0xA708C6A6)
    fun sceAudioGetChannelRestLength(cpu: CpuState): Unit = UNIMPLEMENTED(0xB011922F)
    fun sceAudioSRCOutputBlocking(cpu: CpuState): Unit = UNIMPLEMENTED(0xE0727056)
    fun sceAudioOutputPanned(cpu: CpuState): Unit = UNIMPLEMENTED(0xE2D56B2D)
    fun sceAudioInputInitEx(cpu: CpuState): Unit = UNIMPLEMENTED(0xE926D3FB)
    fun sceAudioGetChannelRestLen(cpu: CpuState): Unit = UNIMPLEMENTED(0xE9D97901)


    override fun registerModule() {
        registerFunctionInt("sceAudioChReserve", 0x5EC81C55, since = 150) { sceAudioChReserve(int, int, int) }
        registerFunctionInt("sceAudioChangeChannelVolume", 0xB7E1D8E7, since = 150) {
            sceAudioChangeChannelVolume(
                int,
                int,
                int
            )
        }
        registerFunctionSuspendInt(
            "sceAudioOutputPannedBlocking",
            0x13F592BC,
            since = 150
        ) { sceAudioOutputPannedBlocking(int, int, int, int) }
        registerFunctionSuspendInt("sceAudioOutputBlocking", 0x136CAF51, since = 150) {
            sceAudioOutputBlocking(
                int,
                int,
                int
            )
        }
        registerFunctionInt("sceAudioSetChannelDataLen", 0xCB2E439E, since = 150) {
            sceAudioSetChannelDataLen(
                int,
                int
            )
        }
        registerFunctionInt("sceAudioChangeChannelConfig", 0x95FD0C2D, since = 150) {
            sceAudioChangeChannelConfig(
                int,
                int
            )
        }

        registerFunctionRaw("sceAudioOutput2Reserve", 0x01562BA3, since = 150) { sceAudioOutput2Reserve(it) }
        registerFunctionRaw("sceAudioInputBlocking", 0x086E5895, since = 150) { sceAudioInputBlocking(it) }
        registerFunctionRaw(
            "sceAudioOutput2OutputBlocking",
            0x2D53F36E,
            since = 150
        ) { sceAudioOutput2OutputBlocking(it) }
        registerFunctionRaw("sceAudioSRCChReserve", 0x38553111, since = 150) { sceAudioSRCChReserve(it) }
        registerFunctionRaw("sceAudioOneshotOutput", 0x41EFADE7, since = 150) { sceAudioOneshotOutput(it) }
        registerFunctionRaw("sceAudioOutput2Release", 0x43196845, since = 150) { sceAudioOutput2Release(it) }
        registerFunctionRaw("sceAudioSRCChRelease", 0x5C37C0AE, since = 150) { sceAudioSRCChRelease(it) }
        registerFunctionRaw("sceAudioOutput2ChangeLength", 0x63F2889C, since = 150) { sceAudioOutput2ChangeLength(it) }
        registerFunctionRaw(
            "sceAudioOutput2GetRestSample",
            0x647CEF33,
            since = 150
        ) { sceAudioOutput2GetRestSample(it) }
        registerFunctionRaw("sceAudioInput", 0x6D4BEC68, since = 150) { sceAudioInput(it) }
        registerFunctionRaw("sceAudioChRelease", 0x6FC46853, since = 150) { sceAudioChRelease(it) }
        registerFunctionRaw("sceAudioInputInit", 0x7DE61688, since = 150) { sceAudioInputInit(it) }
        registerFunctionRaw("sceAudioWaitInputEnd", 0x87B2E651, since = 150) { sceAudioWaitInputEnd(it) }
        registerFunctionRaw("sceAudioOutput", 0x8C1009B2, since = 150) { sceAudioOutput(it) }
        registerFunctionRaw("sceAudioPollInputEnd", 0xA633048E, since = 150) { sceAudioPollInputEnd(it) }
        registerFunctionRaw("sceAudioGetInputLength", 0xA708C6A6, since = 150) { sceAudioGetInputLength(it) }
        registerFunctionRaw(
            "sceAudioGetChannelRestLength",
            0xB011922F,
            since = 150
        ) { sceAudioGetChannelRestLength(it) }
        registerFunctionRaw("sceAudioSRCOutputBlocking", 0xE0727056, since = 150) { sceAudioSRCOutputBlocking(it) }
        registerFunctionRaw("sceAudioOutputPanned", 0xE2D56B2D, since = 150) { sceAudioOutputPanned(it) }
        registerFunctionRaw("sceAudioInputInitEx", 0xE926D3FB, since = 150) { sceAudioInputInitEx(it) }
        registerFunctionRaw("sceAudioGetChannelRestLen", 0xE9D97901, since = 150) { sceAudioGetChannelRestLen(it) }
    }
}
