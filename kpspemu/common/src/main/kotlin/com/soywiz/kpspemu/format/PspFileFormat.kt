package com.soywiz.kpspemu.format

import com.soywiz.korio.file.*
import com.soywiz.korio.lang.*
import com.soywiz.korio.stream.*

enum class PspFileFormat(val fileContainer: Boolean = false) {
    ELF, ENCRYPTED_ELF, PBP,
    CSO(fileContainer = true),
    ISO(fileContainer = true),
    ZIP(fileContainer = true),
    ;

    companion object {
        suspend fun detect(file: VfsFile): PspFileFormat? = detect(file.open(), file.basename)
        suspend fun detect(stream: AsyncStream, name: String = "unknown.bin"): PspFileFormat? =
            detect(stream.duplicate().readBytesExact(4).openSync(), name)

        fun detect(header: ByteArray, name: String = "unknown.bin"): PspFileFormat? = detect(header.openSync(), name)
        fun detect(stream: SyncStream, name: String = "unknown.bin"): PspFileFormat? {
            val magicId = stream.clone().readString(4, ASCII)
            val file = PathInfo(name)
            //println("magicId == '$magicId'")
            return when {
                magicId == "\u007fELF" -> ELF
                magicId == "\u007ePSP" -> ENCRYPTED_ELF
                magicId == "\u0000PBP" -> PBP
                magicId == "PK\u0003\u0004" -> ZIP
                magicId == "CISO" -> CSO
                file.extensionLC == "iso" -> ISO
                magicId == "\u0000\u0000\u0000\u0000" -> ISO // CD001 at sector 0x10
                else -> null
            }
        }
    }
}

fun ByteArray.detectPspFormat() = PspFileFormat.detect(this)
suspend fun VfsFile.detectPspFormat() = PspFileFormat.detect(this)
suspend fun AsyncStream.detectPspFormat(name: String = "unknown.bin") = PspFileFormat.detect(this, name)
