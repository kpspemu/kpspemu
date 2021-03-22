package com.soywiz.kpspemu.util.io

import com.soywiz.korio.file.*
import com.soywiz.korio.file.std.*
import com.soywiz.korio.stream.*

private fun VfsFile.interceptPsp(): VfsFile {
    val SCE_LBN_REGEX = Regex("^/?sce_lbn0x([0-9a-fA-F]+)_size0x([0-9a-fA-F]+)$")

    val base = this
    val isoVfs = this.vfs as IsoVfs
    return VfsFile(object : Vfs.Decorator(base) {
        override suspend fun open(path: String, mode: VfsOpenMode): AsyncStream {
            println("Opening ISO path: $path")
            val result = SCE_LBN_REGEX.matchEntire(path)
            return if (result != null) {
                val (_, lbnString, sizeString) = result.groupValues
                val lbn = lbnString.toInt(16)
                val size = sizeString.toInt(16)
                println("Matching sce_lbn: ${result.groupValues} : $lbn, $size")
                isoVfs.isoFile.reader.getSector(lbn, size)
            } else {
                base[path].open(mode)
            }
        }
    }, base.path)
}

suspend fun PspIsoVfs(file: VfsFile): VfsFile = ISO.openVfs(file.open(VfsOpenMode.READ), true).interceptPsp()
suspend fun PspIsoVfs(s: AsyncStream): VfsFile = ISO.openVfs(s, true).interceptPsp()
suspend fun AsyncStream.openAsPspIso(): VfsFile = PspIsoVfs(this)
suspend fun VfsFile.openAsPspIso() = PspIsoVfs(this)
