package com.soywiz.kpspemu.hle.manager

import com.soywiz.korio.async.*
import com.soywiz.korio.file.*
import com.soywiz.korio.file.VfsUtil
import com.soywiz.korio.lang.*
import com.soywiz.korio.stream.*
import com.soywiz.kpspemu.*
import com.soywiz.kpspemu.util.*
import kotlinx.coroutines.*

class FileManager(val emulator: Emulator) {
    companion object {
        val INIT_CURRENT_DIRECTORY = "umd0:"
        val INIT_EXECUTABLE_FILE = "umd0:/PSP_GAME/USRDIR/EBOOT.BIN"
    }

    val deviceManager get() = emulator.deviceManager
    var currentDirectory = INIT_CURRENT_DIRECTORY
    var executableFile = INIT_EXECUTABLE_FILE

    val fileDescriptors = ResourceList<FileDescriptor>("FileDescriptor") { FileDescriptor(it) }
    val directoryDescriptors = ResourceList<DirectoryDescriptor>("DirectoryDescriptor") { DirectoryDescriptor(it) }

    fun reset() {
        currentDirectory = INIT_CURRENT_DIRECTORY
        executableFile = INIT_EXECUTABLE_FILE
        fileDescriptors.reset()
        directoryDescriptors.reset()
    }

    fun resolvePath(path: String): String {
        if (path.contains(':')) {
            return path
        } else {
            if (path.startsWith('/')) {
                return currentDirectory.split(':').first() + ":" + path
            } else {
                return VfsUtil.combine(currentDirectory, path)
            }
        }
    }

    fun resolve(path: String): VfsFile {
        val resolvedPath = resolvePath(path)
        //println("resolvedPath --> $resolvedPath")
        return deviceManager.root[resolvedPath]
    }
}

class FileDescriptor(override val id: Int) : ResourceItem {
    lateinit var fileName: String
    lateinit var file: VfsFile
    lateinit var stream: AsyncStream

    var doLater: (suspend () -> Unit)? = null
    var asyncPromise: Deferred<Unit>? = null
    var asyncResult: Long = 0L
    var asyncDone: Boolean = false
}

class DirectoryDescriptor(override val id: Int) : ResourceItem {
    lateinit var directory: VfsFile
    var pos: Int = 0
    var files: List<VfsFile> = listOf()
    val remaining: Int get() = files.size - pos
}

data class SceIoStat(
    var mode: Int = 0, // SceMode
    var attributes: Int = 0, // IOFileModes.File
    var size: Long = 0L,
    var timeCreation: ScePspDateTime = ScePspDateTime(0L),
    var timeLastAccess: ScePspDateTime = ScePspDateTime(0L),
    var timeLastModification: ScePspDateTime = ScePspDateTime(0L),
    var device: IntArray = IntArray(6)
) {
    companion object : Struct<SceIoStat>(
        { SceIoStat() },
        SceIoStat::mode AS INT32,
        SceIoStat::attributes AS INT32,
        SceIoStat::size AS INT64,
        SceIoStat::timeCreation AS ScePspDateTime,
        SceIoStat::timeLastAccess AS ScePspDateTime,
        SceIoStat::timeLastModification AS ScePspDateTime,
        SceIoStat::device AS INTLIKEARRAY(INT32, 6)
    )
}

//class SceIoStat(
//	val mode: Int,
//	val attributes: Int,
//	val size: Long,
//	val timeCreation: ScePspDateTime,
//	val timeLastAccess: ScePspDateTime,
//	val timeLastModifications: ScePspDateTime,
//	val device: IntArray = IntArray(6)
//) {
//	fun write(s: SyncStream) = s.run {
//		write32_le(mode)
//		write32_le(attributes)
//		write64_le(size)
//		timeCreation.write(s)
//		timeLastAccess.write(s)
//		timeLastModifications.write(s)
//		for (n in 0 until 6) write32_le(device[n])
//	}
//}


data class HleIoDirent(
    var stat: SceIoStat = SceIoStat(),
    var name: String = "",
    var privateData: Int = 0,
    var dummy: Int = 0
) {
    companion object : Struct<HleIoDirent>(
        { HleIoDirent() },
        HleIoDirent::stat AS SceIoStat,
        HleIoDirent::name AS STRINGZ(UTF8, 256),
        HleIoDirent::privateData AS INT32,
        HleIoDirent::dummy AS INT32
    )
}

object IOFileModes {
    val DIR = 0x1000
    val FILE = 0x2000

    val FormatMask = 0x0038
    val SymbolicLink = 0x0008
    val Directory = 0x0010
    val File = 0x0020
    val CanRead = 0x0004
    val CanWrite = 0x0002
    val CanExecute = 0x0001
}

object SeekType {
    val Set = 0
    val Cur = 1
    val End = 2
    val Tell = 65536
}

object FileOpenFlags {
    val Read = 0x0001
    val Write = 0x0002
    val ReadWrite = Read or Write
    val NoBlock = 0x0004
    val _InternalDirOpen = 0x0008 // Internal use for dopen
    val Append = 0x0100
    val Create = 0x0200
    val Truncate = 0x0400
    val Excl = 0x0800
    val Unknown1 = 0x4000 // something async?
    val NoWait = 0x8000
    val Unknown2 = 0xf0000 // seen on Wipeout Pure and Infected
    val Unknown3 = 0x2000000 // seen on Puzzle Guzzle, Hammerin' Hero
}

//object IOFileModes {
//	val FormatMask = 0x0038
//	val SymbolicLink = 0x0008
//	val Directory = 0x0010
//	val File = 0x0020
//	val CanRead = 0x0004
//	val CanWrite = 0x0002
//	val CanExecute = 0x0001
//}