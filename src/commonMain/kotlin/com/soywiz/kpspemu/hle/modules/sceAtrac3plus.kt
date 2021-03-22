package com.soywiz.kpspemu.hle.modules

import com.soywiz.kmem.*
import com.soywiz.korau.format.atrac3plus.*
import com.soywiz.korau.format.atrac3plus.Atrac3plusDecoder.Companion.ATRAC3P_FRAME_SAMPLES
import com.soywiz.korau.format.atrac3plus.util.*
import com.soywiz.korau.format.atrac3plus.util.Atrac3PlusUtil.PSP_CODEC_AT3
import com.soywiz.korau.format.atrac3plus.util.Atrac3PlusUtil.PSP_CODEC_AT3PLUS
import com.soywiz.korio.lang.*
import com.soywiz.kpspemu.*
import com.soywiz.kpspemu.cpu.*
import com.soywiz.kpspemu.hle.*
import com.soywiz.kpspemu.hle.error.*
import com.soywiz.kpspemu.mem.*

@Suppress("UNUSED_PARAMETER")
class sceAtrac3plus(emulator: Emulator) :
    SceModule(emulator, "sceAtrac3plus", 0x00010011, "libatrac3plus.prx", "sceATRAC3plus_Library") {
    class AtracID(val id: Int) {
        val decoder = Atrac3plusDecoder()
        var inputBuffer = PtrArray(DummyPtr, 0)
        var inUse = false
        var isSecondBufferNeeded = false
        var isSecondBufferSet = false
        val info = Atrac3PlusUtil.AtracFileInfo()
        var atracCurrentSample: Int = 0
        val atracEndSample: Int get() = info.atracEndSample
        val secondBufferReadPosition: Int = 0
        val secondBufferSize: Int = 0
        var readAddr: Int = 0
        val remainFrames: Int get() = 0 // @TODO
    }

    private val atracIDs = Array(6) { AtracID(it) }

    fun getAtrac(id: Int) = atracIDs[id]

    fun getStartSkippedSamples(codecType: Int) = when (codecType) {
        PSP_CODEC_AT3 -> 69
        PSP_CODEC_AT3PLUS -> 368
        else -> 0
    }

    fun getMaxSamples(codecType: Int) = when (codecType) {
        PSP_CODEC_AT3 -> 1024
        PSP_CODEC_AT3PLUS -> ATRAC3P_FRAME_SAMPLES
        else -> 0
    }

    fun sceAtracSetDataAndGetID(data: Ptr, bufferSize: Int): Int {
        val id = atracIDs.firstOrNull { !it.inUse } ?: sceKernelException(SceKernelErrors.ERROR_ATRAC_NO_ID)
        val info = id.info
        logger.trace { "sceAtracSetDataAndGetID Partially implemented" }
        val res = Atrac3PlusUtil.analyzeRiffFile(data.openSync(), 0, bufferSize, id.info)
        if (res < 0) return res
        val outputChannels = 2
        id.inputBuffer = PtrArray(data, bufferSize)

        val startSkippedSamples = getStartSkippedSamples(PSP_CODEC_AT3PLUS)
        val maxSamples = getMaxSamples(PSP_CODEC_AT3PLUS)
        val skippedSamples = startSkippedSamples + info.atracSampleOffset
        val skippedFrames = (skippedSamples.toDouble() / maxSamples.toDouble()).toIntCeil()

        id.readAddr = data.addr + id.info.inputFileDataOffset + (skippedFrames * info.atracBytesPerFrame)
        id.decoder.init(id.info.atracBytesPerFrame, id.info.atracChannels, outputChannels, 0)
        return id.id
    }

    fun sceAtracGetSecondBufferInfo(atID: Int, puiPosition: Ptr32, puiDataByte: Ptr32): Int {
        logger.error { "sceAtracGetSecondBufferInfo Not implemented ($atID, $puiPosition, $puiDataByte)" }
        val id = getAtrac(atID)
        if (!id.isSecondBufferNeeded) {
            puiPosition[0] = 0
            puiDataByte[0] = 0
            return SceKernelErrors.ERROR_ATRAC_SECOND_BUFFER_NOT_NEEDED
        }
        puiPosition[0] = id.secondBufferReadPosition
        puiDataByte[0] = id.secondBufferSize
        return 0
    }

    fun sceAtracSetSecondBuffer(id: Int, puiPosition: Ptr, puiDataByte: Ptr): Int {
        logger.error { "sceAtracSetSecondBuffer Not implemented ($id, $puiPosition, $puiDataByte)" }
        return 0
    }

    fun sceAtracGetSoundSample(id: Int, endSamplePtr: Ptr, loopStartSamplePtr: Ptr, loopEndSamplePtr: Ptr): Int {
        logger.error { "sceAtracGetSoundSample Not implemented ($id, $endSamplePtr, $loopStartSamplePtr, $loopEndSamplePtr)" }
        return 0
    }

    fun sceAtracSetLoopNum(id: Int, numberOfLoops: Int): Int {
        logger.error { "sceAtracSetLoopNum Not implemented ($id, $numberOfLoops)" }
        return 0
    }

    fun sceAtracGetRemainFrame(atID: Int, remainFramePtr: Ptr32): Int {
        logger.error { "sceAtracGetRemainFrame Not implemented ($atID, $remainFramePtr)" }
        val id = getAtrac(atID)
        remainFramePtr.set(-1)
        return 0
    }

    fun sceAtracGetNextDecodePosition(atracId: Int, samplePositionPtr: Ptr): Int {
        val id = getAtrac(atracId)
        logger.info { "sceAtracGetNextDecodePosition Not implemented ($id, $samplePositionPtr)" }
        if (id.atracCurrentSample >= id.atracEndSample) sceKernelException(SceKernelErrors.ERROR_ATRAC_ALL_DATA_DECODED)
        samplePositionPtr.sw(0, id.atracCurrentSample)
        logger.trace { "sceAtracGetNextDecodePosition returning pos=%d".format(samplePositionPtr.lw(0)) }

        return 0
    }

    fun sceAtracDecodeData(
        idAT: Int,
        samplesAddr: Ptr,
        samplesNbrAddr: Ptr32,
        outEndAddr: Ptr,
        remainFramesAddr: Ptr32
    ): Int {
        logger.trace { "sceAtracDecodeData Not implemented ($idAT, $samplesAddr, $samplesNbrAddr, $outEndAddr, $remainFramesAddr)" }
        val id = getAtrac(idAT)
        val info = id.info
        if (id.isSecondBufferNeeded && !id.isSecondBufferSet) {
            logger.warn { "sceAtracDecodeData atracID=0x%X needs second buffer!".format(idAT) }
            return SceKernelErrors.ERROR_ATRAC_SECOND_BUFFER_NEEDED
        }

        //val result = id.decoder.decode(emulator.imem, samplesAddr.addr, outEndAddr)
        val result = id.decoder.decode(emulator.imem, id.readAddr, id.info.atracBytesPerFrame, samplesAddr.openSync())
        if (result < 0) {
            samplesNbrAddr.set(0)
            return result
        }

        id.readAddr += info.atracBytesPerFrame
        samplesNbrAddr.set(id.decoder.numberOfSamples)
        //remainFramesAddr.set(id.remainFrames)
        remainFramesAddr.set(id.remainFrames)

        if (result == 0) threadManager.delayThread(2300)

        return result
    }

    fun sceAtracGetStreamDataInfo(
        idAT: Int,
        writePointerPointer: Ptr,
        availableBytesPtr: Ptr,
        readOffsetPtr: Ptr
    ): Int {
        logger.error { "sceAtracGetStreamDataInfo Not implemented ($idAT, $writePointerPointer, $availableBytesPtr, $readOffsetPtr)" }
        val id = getAtrac(idAT)

        //if (inputBuffer.getFileWriteSize() <= 0 && currentLoopNum >= 0 && info.loopNum != 0) {
        //	// Read ahead to restart the loop
        //	inputBuffer.setFilePosition(getFilePositionFromSample(info.loops[currentLoopNum].startSample))
        //	reloadingFromLoopStart = true
        //}
//
        //// Remember the CurrentSample at the time of the getStreamDataInfo
        //getStreamDataInfoCurrentSample = getAtracCurrentSample()
//
        //writeAddr.setValue(inputBuffer.getWriteAddr())
        //writableBytesAddr.setValue(inputBuffer.getWriteSize())
        //readOffsetAddr.setValue(inputBuffer.getFilePosition())

        return 0
    }

    fun sceAtracAddStreamData(id: Int, bytesToAdd: Int): Int {
        return 0
    }

    fun sceAtracReleaseAtracID(id: Int): Int {
        val atrac = getAtrac(id)
        atrac.inUse = false
        return 0
    }

    fun sceAtracGetAtracID(codecType: Int): Int {
        logger.warn { "sceAtracGetAtracID not implemented" }
        return 1
    }

    fun sceAtracSetData(cpu: CpuState): Unit = UNIMPLEMENTED(0x0E2A73AB)
    fun sceAtracSetHalfwayBufferAndGetID(cpu: CpuState): Unit = UNIMPLEMENTED(0x0FAE370E)
    fun sceAtracReinit(cpu: CpuState): Unit = UNIMPLEMENTED(0x132F1ECA)
    fun sceAtrac3plus_2DD3E298(cpu: CpuState): Unit = UNIMPLEMENTED(0x2DD3E298)
    fun sceAtracGetChannel(cpu: CpuState): Unit = UNIMPLEMENTED(0x31668BAA)
    fun sceAtracGetNextSample(cpu: CpuState): Unit = UNIMPLEMENTED(0x36FAABFB)
    fun sceAtracSetHalfwayBuffer(cpu: CpuState): Unit = UNIMPLEMENTED(0x3F6E26B5)
    fun sceAtracSetAA3DataAndGetID(cpu: CpuState): Unit = UNIMPLEMENTED(0x5622B7C1)
    fun sceAtracSetMOutHalfwayBuffer(cpu: CpuState): Unit = UNIMPLEMENTED(0x5CF9D852)
    fun sceAtracSetAA3HalfwayBufferAndGetID(cpu: CpuState): Unit = UNIMPLEMENTED(0x5DD66588)
    fun sceAtracResetPlayPosition(cpu: CpuState): Unit = UNIMPLEMENTED(0x644E5607)
    fun sceAtracSetMOutHalfwayBufferAndGetID(cpu: CpuState): Unit = UNIMPLEMENTED(0x9CD7DE03)
    fun sceAtracGetBitrate(cpu: CpuState): Unit = UNIMPLEMENTED(0xA554A158)
    fun sceAtracGetOutputChannel(cpu: CpuState): Unit = UNIMPLEMENTED(0xB3B5D042)
    fun sceAtracGetBufferInfoForReseting(cpu: CpuState): Unit = UNIMPLEMENTED(0xCA3CA3D2)
    fun sceAtracStartEntry(cpu: CpuState): Unit = UNIMPLEMENTED(0xD1F59FDB)
    fun sceAtracEndEntry(cpu: CpuState): Unit = UNIMPLEMENTED(0xD5C28CC0)
    fun sceAtracGetMaxSample(cpu: CpuState): Unit = UNIMPLEMENTED(0xD6A5F2F7)
    fun sceAtracGetInternalErrorInfo(cpu: CpuState): Unit = UNIMPLEMENTED(0xE88F759B)
    fun sceAtracIsSecondBufferNeeded(cpu: CpuState): Unit = UNIMPLEMENTED(0xECA32A99)
    fun sceAtracGetLoopStatus(cpu: CpuState): Unit = UNIMPLEMENTED(0xFAA4F89B)


    override fun registerModule() {
        registerFunctionInt("sceAtracSetDataAndGetID", 0x7A20E7AF, since = 150) { sceAtracSetDataAndGetID(ptr, int) }
        registerFunctionInt("sceAtracReleaseAtracID", 0x61EB33F5, since = 150) { sceAtracReleaseAtracID(int) }
        registerFunctionInt("sceAtracGetSecondBufferInfo", 0x83E85EA0, since = 150) {
            sceAtracGetSecondBufferInfo(
                int,
                ptr32,
                ptr32
            )
        }
        registerFunctionInt("sceAtracSetSecondBuffer", 0x83BF7AFD, since = 150) {
            sceAtracSetSecondBuffer(
                int,
                ptr,
                ptr
            )
        }
        registerFunctionInt("sceAtracGetSoundSample", 0xA2BBA8BE, since = 150) {
            sceAtracGetSoundSample(
                int,
                ptr,
                ptr,
                ptr
            )
        }
        registerFunctionInt("sceAtracSetLoopNum", 0x868120B5, since = 150) { sceAtracSetLoopNum(int, int) }
        registerFunctionInt("sceAtracGetRemainFrame", 0x9AE849A7, since = 150) { sceAtracGetRemainFrame(int, ptr32) }
        registerFunctionInt("sceAtracGetNextDecodePosition", 0xE23E3A35, since = 150) {
            sceAtracGetNextDecodePosition(
                int,
                ptr
            )
        }
        registerFunctionInt("sceAtracDecodeData", 0x6A8C3CD5, since = 150) {
            sceAtracDecodeData(
                int,
                ptr,
                ptr32,
                ptr,
                ptr32
            )
        }
        registerFunctionInt("sceAtracGetStreamDataInfo", 0x5D268707, since = 150) {
            sceAtracGetStreamDataInfo(
                int,
                ptr,
                ptr,
                ptr
            )
        }
        registerFunctionInt("sceAtracAddStreamData", 0x7DB31251, since = 150) { sceAtracAddStreamData(int, int) }
        registerFunctionInt("sceAtracGetAtracID", 0x780F88D1, since = 150) { sceAtracGetAtracID(int) }

        registerFunctionRaw("sceAtracSetData", 0x0E2A73AB, since = 150) { sceAtracSetData(it) }
        registerFunctionRaw(
            "sceAtracSetHalfwayBufferAndGetID",
            0x0FAE370E,
            since = 150
        ) { sceAtracSetHalfwayBufferAndGetID(it) }
        registerFunctionRaw("sceAtracReinit", 0x132F1ECA, since = 150) { sceAtracReinit(it) }
        registerFunctionRaw("sceAtrac3plus_2DD3E298", 0x2DD3E298, since = 150) { sceAtrac3plus_2DD3E298(it) }
        registerFunctionRaw("sceAtracGetChannel", 0x31668BAA, since = 150) { sceAtracGetChannel(it) }
        registerFunctionRaw("sceAtracGetNextSample", 0x36FAABFB, since = 150) { sceAtracGetNextSample(it) }
        registerFunctionRaw("sceAtracSetHalfwayBuffer", 0x3F6E26B5, since = 150) { sceAtracSetHalfwayBuffer(it) }
        registerFunctionRaw("sceAtracSetAA3DataAndGetID", 0x5622B7C1, since = 150) { sceAtracSetAA3DataAndGetID(it) }
        registerFunctionRaw(
            "sceAtracSetMOutHalfwayBuffer",
            0x5CF9D852,
            since = 150
        ) { sceAtracSetMOutHalfwayBuffer(it) }
        registerFunctionRaw(
            "sceAtracSetAA3HalfwayBufferAndGetID",
            0x5DD66588,
            since = 150
        ) { sceAtracSetAA3HalfwayBufferAndGetID(it) }
        registerFunctionRaw("sceAtracResetPlayPosition", 0x644E5607, since = 150) { sceAtracResetPlayPosition(it) }
        registerFunctionRaw(
            "sceAtracSetMOutHalfwayBufferAndGetID",
            0x9CD7DE03,
            since = 150
        ) { sceAtracSetMOutHalfwayBufferAndGetID(it) }
        registerFunctionRaw("sceAtracGetBitrate", 0xA554A158, since = 150) { sceAtracGetBitrate(it) }
        registerFunctionRaw("sceAtracGetOutputChannel", 0xB3B5D042, since = 150) { sceAtracGetOutputChannel(it) }
        registerFunctionRaw(
            "sceAtracGetBufferInfoForReseting",
            0xCA3CA3D2,
            since = 150
        ) { sceAtracGetBufferInfoForReseting(it) }
        registerFunctionRaw("sceAtracStartEntry", 0xD1F59FDB, since = 150) { sceAtracStartEntry(it) }
        registerFunctionRaw("sceAtracEndEntry", 0xD5C28CC0, since = 150) { sceAtracEndEntry(it) }
        registerFunctionRaw("sceAtracGetMaxSample", 0xD6A5F2F7, since = 150) { sceAtracGetMaxSample(it) }
        registerFunctionRaw(
            "sceAtracGetInternalErrorInfo",
            0xE88F759B,
            since = 150
        ) { sceAtracGetInternalErrorInfo(it) }
        registerFunctionRaw(
            "sceAtracIsSecondBufferNeeded",
            0xECA32A99,
            since = 150
        ) { sceAtracIsSecondBufferNeeded(it) }
        registerFunctionRaw("sceAtracGetLoopStatus", 0xFAA4F89B, since = 150) { sceAtracGetLoopStatus(it) }
    }
}
