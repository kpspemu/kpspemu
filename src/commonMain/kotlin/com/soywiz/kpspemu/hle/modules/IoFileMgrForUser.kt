package com.soywiz.kpspemu.hle.modules

import com.soywiz.kmem.*
import com.soywiz.korio.async.*
import com.soywiz.korio.error.*
import com.soywiz.korio.file.*
import com.soywiz.korio.lang.*
import com.soywiz.korio.stream.*
import com.soywiz.kpspemu.*
import com.soywiz.kpspemu.cpu.*
import com.soywiz.kpspemu.hle.*
import com.soywiz.kpspemu.hle.error.*
import com.soywiz.kpspemu.hle.manager.*
import com.soywiz.kpspemu.mem.*
import com.soywiz.kpspemu.util.*
import kotlinx.coroutines.*
import kotlinx.coroutines.channels.*
import kotlinx.coroutines.flow.*
import kotlin.math.*
import com.soywiz.korio.lang.invalidOp as invalidOp1

@Suppress("UNUSED_PARAMETER")
class IoFileMgrForUser(emulator: Emulator) :
    SceModule(emulator, "IoFileMgrForUser", 0x40010011, "iofilemgr.prx", "sceIOFileManager") {
    companion object {
        const val EMULATOR_DEVCTL__GET_HAS_DISPLAY = 0x00000001
        const val EMULATOR_DEVCTL__SEND_OUTPUT = 0x00000002
        const val EMULATOR_DEVCTL__IS_EMULATOR = 0x00000003
        const val EMULATOR_DEVCTL__SEND_CTRLDATA = 0x00000010
        const val EMULATOR_DEVCTL__EMIT_SCREENSHOT = 0x00000020
    }

    val fileDescriptors get() = emulator.fileManager.fileDescriptors
    val directoryDescriptors get() = emulator.fileManager.directoryDescriptors

    private fun _resolve(path: String): VfsFile {
        val resolved = fileManager.resolve(path)
        logger.trace { "resolved:$resolved" }
        return resolved
    }

    private fun String.normalizePath(): String = this.replace('\\', '/').replace("./", "/").replace("//", "/")

    private fun resolve(path: String?): VfsFile {
        val npath = path?.normalizePath()
        logger.trace { "resolve:$path" }
        logger.trace { "resolveNormalized:$npath" }
        return _resolve(npath!!)
    }

    suspend fun _sceIoOpen(thread: PspThread, fileId: Int, fileName: String?, flags: Int, mode: Int): Int {
        logger.warn { "WIP: _sceIoOpen(${thread.name}): $fileId, $fileName, $flags, $mode" }
        logger.warn { " --> normalized=  ${fileName?.normalizePath()}" }
        if (fileName == null) return SceKernelErrors.ERROR_ERROR
        try {
            val file = fileDescriptors[fileId]
            file.fileName = fileName
            file.file = resolve(fileName)
            val flags2 = when {
                (flags and FileOpenFlags.Truncate) != 0 -> VfsOpenMode.CREATE_OR_TRUNCATE
                (flags and FileOpenFlags.Create) != 0 -> VfsOpenMode.CREATE
                (flags and FileOpenFlags.Write) != 0 -> VfsOpenMode.WRITE
                else -> VfsOpenMode.READ
            }
            //val f = file.file.open(flags2)
            //val bytes = f.readAll()
            //println("Bytes:" + bytes.size)
            //file.stream = bytes.openAsync().check(fileName)

            //file.stream = file.file.open(flags2).cached(fileName)
            file.stream = file.file.open(flags2)
            logger.warn { "WIP: sceIoOpen(${thread.name}) --> $fileName, ${file.id}" }
            return file.id
        } catch (e: Throwable) {
            println("Error openingfile(${thread.name}): $fileName : '${e.message}'")
            //e.printStackTrace()
            logger.warn { "   --> ${SceKernelErrors.ERROR_ERRNO_FILE_NOT_FOUND}" }
            return SceKernelErrors.ERROR_ERRNO_FILE_NOT_FOUND
        }
    }

    suspend fun sceIoOpen(thread: PspThread, fileName: String?, flags: Int, mode: Int): Int {
        return _sceIoOpen(thread, fileDescriptors.alloc().id, fileName, flags, mode)
    }

    fun VfsStat.toSce() = SceIoStat(
        mode = 511,
        attributes = when {
            isDirectory -> IOFileModes.DIR or IOFileModes.Directory or IOFileModes.CanRead
            else -> IOFileModes.FILE or IOFileModes.File or IOFileModes.CanRead or IOFileModes.CanExecute or IOFileModes.CanWrite
        },
        size = size,
        timeCreation = ScePspDateTime(createDate),
        timeLastAccess = ScePspDateTime(lastAccessDate),
        timeLastModification = ScePspDateTime(modifiedDate),
        device = (extraInfo as? IntArray?) ?: intArrayOf(0, 0, 0, 0, 0, 0)
    )

    suspend fun sceIoGetstat(fileName: String?, ptr: Ptr): Int {
        logger.warn { "sceIoGetstat:$fileName,$ptr" }
        val file = resolve(fileName)
        val fstat = file.stat()
        var result: Int
        if (fstat.exists) {
            val stat = fstat.toSce()
            if (ptr.isNotNull) {
                ptr.openSync().write(SceIoStat, stat)
            }
            result = 0
        } else {
            result = SceKernelErrors.ERROR_ERRNO_FILE_NOT_FOUND
        }
        logger.warn { "sceIoGetstat --> $result" }
        return result
    }

    suspend fun sceIoLseek32(fileId: Int, offset: Int, whence: Int): Int {
        return _sceIoLseek(fileId, offset.toLong(), whence).toInt()
    }

    suspend fun sceIoLseek(fileId: Int, offset: Long, whence: Int): Long {
        return _sceIoLseek(fileId, offset, whence)
    }

    suspend fun _sceIoLseek(fileId: Int, offset: Long, whence: Int): Long {
        logger.trace { "WIP: _sceIoLseek: $fileId, $offset, $whence" }
        val stream = fileDescriptors[fileId].stream
        stream.position = when (whence) {
            SeekType.Set -> offset
            SeekType.Cur -> stream.position + offset
            SeekType.End -> stream.size() + offset
            SeekType.Tell -> stream.position
            else -> invalidOp1("Invalid sceIoLseek32")
        }
        return stream.position
    }


    suspend fun sceIoRead(fileId: Int, dst: Ptr, dstLen: Int): Int {
        logger.info { "WIP: sceIoRead: $fileId, $dst, $dstLen" }
        val fd = fileDescriptors[fileId]
        val stream = fd.stream
        val adstLen = max(0, dstLen)
        val out = ByteArray(adstLen)
        val initPosition = stream.position
        val read = stream.read(out, 0, adstLen)
        logger.info { " --> $fileId, $dst, $dstLen : POS($initPosition -> ${stream.position}) LEN(${stream.getLength()}) AVAILABLE(${stream.getAvailable()}) // ${fd.fileName} // read=$read" }
        dst.writeBytes(out, 0, read)
        return read
    }

    suspend fun sceIoWrite(fileId: Int, ptr: Ptr, size: Int): Int {
        logger.warn { "WIP: sceIoWrite: $fileId, $ptr, $size" }
        try {
            //logger.error { "WIP: sceIoWrite: $fileId, $ptr, $size" }
            println("----> " + ptr.readBytes(size).toString(UTF8))

            val stream = fileDescriptors[fileId].stream
            val bytes = ptr.readBytes(size)
            stream.write(bytes)
        } catch (e: Throwable) {
            println("### error writting: ${e.message}")
        }

        return 0
    }

    suspend fun sceIoClose(fileId: Int): Int {
        logger.warn { "WIP: sceIoClose: $fileId" }
        fileDescriptors.freeById(fileId)
        return 0
    }


    suspend fun sceIoDevctl(
        deviceName: String?,
        command: Int,
        inputPointer: Ptr,
        inputLength: Int,
        outputPointer: Ptr,
        outputLength: Int
    ): Int {
        when (deviceName) {
            "kemulator:", "emulator:" -> {
                when (command) {
                    EMULATOR_DEVCTL__IS_EMULATOR -> return 0 // Yes, we are in an emulator!
                    EMULATOR_DEVCTL__GET_HAS_DISPLAY -> {
                        outputPointer.sw(0, display.exposeDisplay.toInt())
                        return 0
                    }
                    EMULATOR_DEVCTL__SEND_OUTPUT -> {
                        emulator.output.append(inputPointer.readBytes(inputLength).toString(UTF8))
                        return 0
                    }
                    EMULATOR_DEVCTL__SEND_CTRLDATA -> {
                        println("EMULATOR_DEVCTL__SEND_CTRLDATA")
                        return 0
                    }
                    EMULATOR_DEVCTL__EMIT_SCREENSHOT -> {
                        println("EMULATOR_DEVCTL__EMIT_SCREENSHOT")
                        return 0
                    }
                    else -> {
                        println("Unhandled emulator command $command")
                        return -1
                    }
                }
            }
        }

        logger.error { "WIP: sceIoDevctl: $deviceName, $command, $inputPointer, $inputLength, $outputPointer, $outputLength" }

        return -1
    }

    suspend fun sceIoDopen(path: String?): Int {
        logger.error { "sceIoDopen:$path" }
        try {
            logger.error { "sceIoDopen:$path" }
            val dd = directoryDescriptors.alloc()
            val dir = resolve(path)
            dd.directory = dir
            dd.pos = 0

            //dd.files = listOf(VfsFile(dir.vfs, dir.fullname + "/."), VfsFile(dir.vfs, dir.fullname + "/..")) + dd.directory.list().toList()
            dd.files = dd.directory.list().toList()
            return dd.id
        } catch (e: Throwable) {
            println("ERROR AT sceIoDopen")
            e.printStackTrace()
            return -1
        }
        //return 0
    }

    suspend fun sceIoDread(id: Int, ptr: Ptr): Int {
        logger.error { "sceIoDread:$id,$ptr" }
        val dd = directoryDescriptors[id]
        if (dd.remaining > 0) {
            val file = dd.files[dd.pos++]
            val stat = file.stat()

            val dirent = HleIoDirent(
                stat = stat.toSce(),
                name = file.basename,
                privateData = 0,
                dummy = 0
            )

            logger.error { "sceIoDread --> $dirent" }

            ptr.openSync().write(HleIoDirent, dirent)
        }
        return dd.remaining
    }

    suspend fun sceIoDclose(id: Int): Int {
        logger.error { "sceIoDclose:$id" }
        directoryDescriptors.freeById(id)
        //return 0
        return 0
    }

    suspend fun sceIoChdir(path: String?): Int {
        logger.error { "sceIoChdir:$path" }
        return 0
    }

    suspend fun sceIoRmdir(path: String?): Int {
        logger.error { "sceIoRmdir:$path" }
        resolve(path).delete()
        return 0
    }

    suspend fun sceIoMkdir(path: String?): Int {
        logger.error { "sceIoMkdir:$path" }
        resolve(path).mkdirsSafe()
        return 0
    }

    class AsyncHandle(
        override val id: Int,
        var promise: Deferred<Unit>? = null,
        var result: Long = 0L,
        var done: Boolean = false
    ) : PoolItem {
        override fun reset() {
            promise = null
            result = 0L
            done = false
        }
    }

    suspend fun asyncv(
        fileId: Int,
        name: String,
        doLater: (suspend () -> Unit)? = null,
        callback: suspend () -> Long
    ): Int {
        logger.error { "starting async $name" }
        val res = fileDescriptors[fileId]
        logger.error { "  STARTED async $name ---> fid=${res.id}" }
        res.asyncDone = false
        res.doLater = doLater
        res.asyncPromise = asyncImmediately(coroutineContext) {
            res.asyncResult = callback()
            logger.error { "  async $name completed with result ${res.asyncResult}" }
            res.asyncDone = true
        }
        return res.id
    }

    suspend fun sceIoOpenAsync(thread: PspThread, filename: String?, flags: Int, mode: Int): Int {
        logger.error { "sceIoOpenAsync:$filename,$flags,$mode" }

        val fid = fileDescriptors.alloc().id

        return asyncv(fid, "sceIoOpenAsync") {
            val res = _sceIoOpen(thread, fid, filename, flags, mode)
            logger.error { "sceIoOpenAsync --> $res" }
            res.toLong()
        }
    }

    suspend fun sceIoReadAsync(fileId: Int, outputPointer: Ptr, outputLength: Int): Int {
        logger.error { "sceIoReadAsync:$fileId,$outputPointer,$outputLength" }
        asyncv(fileId, "sceIoReadAsync") {
            val res = sceIoRead(fileId, outputPointer, outputLength)
            //println(outputPointer.readBytes(outputLength).toString(UTF8))
            logger.error { "::sceIoReadAsync --> $res" }
            res.toLong()
        }
        return 0
    }

    suspend fun sceIoLseekAsync(fileId: Int, offset: Long, whence: Int): Int {
        asyncv(fileId, "sceIoLseekAsync") {
            val res = sceIoLseek(fileId, offset, whence)
            //println(outputPointer.readBytes(outputLength).toString(UTF8))
            logger.error { "::sceIoReadAsync --> $res" }
            res.toLong()
        }
        return 0
    }

    suspend fun sceIoCloseAsync(fileId: Int): Int {
        logger.error { "sceIoCloseAsync:$fileId" }
        asyncv(fileId, "sceIoCloseAsync", doLater = {
            sceIoClose(fileId)
        }) {
            0L
        }
        return 0
    }

    private fun getFileDescriptor(fileId: Int) =
        fileDescriptors.tryGetById(fileId) ?: sceKernelException(SceKernelErrors.ERROR_KERNEL_BAD_FILE_DESCRIPTOR)

    fun sceIoPollAsync(fileId: Int, out: Ptr64): Int {
        logger.error { "sceIoPollAsync:$fileId,$out" }
        val fd = getFileDescriptor(fileId)
        if (out.isNotNull) {
            out.set(fd.asyncResult)
        }
        val outv = when {
            fd.asyncDone -> 0
            else -> 1
        }
        fd.doLater?.let { doLater -> launchImmediately(coroutineContext) { doLater() } } // For closing!
        logger.error { "   sceIoPollAsync:$fileId,$out -> ${fd.asyncResult} -> outv=$outv" }
        return outv
    }

    suspend fun sceIoWaitAsyncCB(fileId: Int, out: Ptr): Int {
        val fd = getFileDescriptor(fileId)
        fd.asyncPromise?.await()
        if (out.isNotNull) {
            out.sdw(0, fd.asyncResult)
        }
        return 0
    }

    fun sceIoGetDevType(cpu: CpuState): Unit = UNIMPLEMENTED(0x08BD7374)
    fun sceIoWriteAsync(cpu: CpuState): Unit = UNIMPLEMENTED(0x0FACAB19)
    fun sceIoLseek32Async(cpu: CpuState): Unit = UNIMPLEMENTED(0x1B385D8F)
    fun sceIoGetFdList(cpu: CpuState): Unit = UNIMPLEMENTED(0x5C2BE2CC)
    fun sceIoIoctl(cpu: CpuState): Unit = UNIMPLEMENTED(0x63632449)
    fun sceIoUnassign(cpu: CpuState): Unit = UNIMPLEMENTED(0x6D08A871)
    fun sceIoRename(cpu: CpuState): Unit = UNIMPLEMENTED(0x779103A0)
    fun sceIoSetAsyncCallback(cpu: CpuState): Unit = UNIMPLEMENTED(0xA12A0514)
    fun sceIoSync(cpu: CpuState): Unit = UNIMPLEMENTED(0xAB96437F)
    fun sceIoChangeAsyncPriority(cpu: CpuState): Unit = UNIMPLEMENTED(0xB293727F)
    fun sceIoChstat(cpu: CpuState): Unit = UNIMPLEMENTED(0xB8A740F4)
    fun sceIoGetAsyncStat(cpu: CpuState): Unit = UNIMPLEMENTED(0xCB05F8D6)
    fun sceIoWaitAsync(cpu: CpuState): Unit = UNIMPLEMENTED(0xE23EEC33)
    fun sceIoCancel(cpu: CpuState): Unit = UNIMPLEMENTED(0xE8BC6571)
    fun sceIoIoctlAsync(cpu: CpuState): Unit = UNIMPLEMENTED(0xE95A012B)

    suspend fun sceIoRemove(path: String?): Int {
        logger.warn { "sceIoRemove('$path') not implemented" }
        return 0
    }

    suspend fun sceIoAssign(dev1: String?, dev2: String?, dev3: String?, mode: Int, unk1: Ptr, unk2: Long): Int {
        logger.warn { "sceIoAssign($dev1, $dev2, $dev3, $mode, $unk1, $unk2) not implemented" }
        return 0
    }


    override fun registerModule() {
        // Devices
        registerFunctionSuspendInt("sceIoDevctl", 0x54F5FB11, since = 150) { sceIoDevctl(str, int, ptr, int, ptr, int) }
        registerFunctionSuspendInt("sceIoAssign", 0xB2A628C1, since = 150) {
            sceIoAssign(
                str,
                str,
                str,
                int,
                ptr,
                long
            )
        }

        // Files
        registerFunctionSuspendInt("sceIoOpen", 0x109F50BC, since = 150) { sceIoOpen(thread, str, int, int) }
        registerFunctionSuspendInt("sceIoLseek32", 0x68963324, since = 150) { sceIoLseek32(int, int, int) }
        registerFunctionSuspendLong("sceIoLseek", 0x27EB27B8, since = 150) { sceIoLseek(int, long, int) }
        registerFunctionSuspendInt("sceIoWrite", 0x42EC03AC, since = 150) { sceIoWrite(int, ptr, int) }
        registerFunctionSuspendInt("sceIoRead", 0x6A638D83, since = 150) { sceIoRead(int, ptr, int) }
        registerFunctionSuspendInt("sceIoClose", 0x810C4BC3, since = 150) { sceIoClose(int) }
        registerFunctionSuspendInt("sceIoGetstat", 0xACE946E8, since = 150) { sceIoGetstat(str, ptr) }
        registerFunctionSuspendInt("sceIoRemove", 0xF27A9C51, since = 150) { sceIoRemove(str) }

        // Files Async
        registerFunctionSuspendInt("sceIoOpenAsync", 0x89AA9906, since = 150) { sceIoOpenAsync(thread, str, int, int) }
        registerFunctionSuspendInt("sceIoReadAsync", 0xA0B5A7C2, since = 150) { sceIoReadAsync(int, ptr, int) }
        registerFunctionSuspendInt("sceIoCloseAsync", 0xFF5940B6, since = 150) { sceIoCloseAsync(int) }
        registerFunctionSuspendInt("sceIoLseekAsync", 0x71B19E77, since = 150) { sceIoLseekAsync(int, long, int) }
        registerFunctionInt("sceIoPollAsync", 0x3251EA56, since = 150) { sceIoPollAsync(int, ptr64) }
        registerFunctionSuspendInt("sceIoWaitAsyncCB", 0x35DBD746, since = 150, cb = true) {
            sceIoWaitAsyncCB(
                int,
                ptr
            )
        }

        // Directories
        registerFunctionSuspendInt("sceIoDopen", 0xB29DDF9C, since = 150) { sceIoDopen(str) }
        registerFunctionSuspendInt("sceIoDread", 0xE3EB004C, since = 150) { sceIoDread(int, ptr) }
        registerFunctionSuspendInt("sceIoDclose", 0xEB092469, since = 150) { sceIoDclose(int) }
        registerFunctionSuspendInt("sceIoMkdir", 0x06A70004, since = 150) { sceIoMkdir(str) }
        registerFunctionSuspendInt("sceIoRmdir", 0x1117C65F, since = 150) { sceIoRmdir(str) }
        registerFunctionSuspendInt("sceIoChdir", 0x55F4717D, since = 150) { sceIoChdir(str) }

        //registerFunctionInt("sceIoDopen", 0xB29DDF9C, since = 150) { sceIoDopen(str) }
        //registerFunctionInt("sceIoDread", 0xE3EB004C, since = 150) { sceIoDread(int, ptr) }
        //registerFunctionInt("sceIoDclose", 0xEB092469, since = 150) { sceIoDclose(int) }
        //registerFunctionInt("sceIoMkdir", 0x06A70004, since = 150) { sceIoMkdir(str) }
        //registerFunctionInt("sceIoRmdir", 0x1117C65F, since = 150) { sceIoRmdir(str) }
        //registerFunctionInt("sceIoChdir", 0x55F4717D, since = 150) { sceIoChdir(str) }

        registerFunctionRaw("sceIoGetDevType", 0x08BD7374, since = 150) { sceIoGetDevType(it) }
        registerFunctionRaw("sceIoWriteAsync", 0x0FACAB19, since = 150) { sceIoWriteAsync(it) }
        registerFunctionRaw("sceIoLseek32Async", 0x1B385D8F, since = 150) { sceIoLseek32Async(it) }
        registerFunctionRaw("sceIoGetFdList", 0x5C2BE2CC, since = 150) { sceIoGetFdList(it) }
        registerFunctionRaw("sceIoIoctl", 0x63632449, since = 150) { sceIoIoctl(it) }
        registerFunctionRaw("sceIoUnassign", 0x6D08A871, since = 150) { sceIoUnassign(it) }
        registerFunctionRaw("sceIoRename", 0x779103A0, since = 150) { sceIoRename(it) }
        registerFunctionRaw("sceIoSetAsyncCallback", 0xA12A0514, since = 150) { sceIoSetAsyncCallback(it) }
        registerFunctionRaw("sceIoSync", 0xAB96437F, since = 150) { sceIoSync(it) }
        registerFunctionRaw("sceIoChangeAsyncPriority", 0xB293727F, since = 150) { sceIoChangeAsyncPriority(it) }
        registerFunctionRaw("sceIoChstat", 0xB8A740F4, since = 150) { sceIoChstat(it) }
        registerFunctionRaw("sceIoGetAsyncStat", 0xCB05F8D6, since = 150) { sceIoGetAsyncStat(it) }
        registerFunctionRaw("sceIoWaitAsync", 0xE23EEC33, since = 150) { sceIoWaitAsync(it) }
        registerFunctionRaw("sceIoCancel", 0xE8BC6571, since = 150) { sceIoCancel(it) }
        registerFunctionRaw("sceIoIoctlAsync", 0xE95A012B, since = 150) { sceIoIoctlAsync(it) }
    }
}

fun AsyncStream.check(name: String): AsyncStream {
    val base = this

    return object : AsyncStreamBase() {
        override suspend fun close() {
            base.close()
        }

        override suspend fun getLength(): Long {
            return base.getLength()
        }

        override suspend fun read(position: Long, buffer: ByteArray, offset: Int, len: Int): Int {
            if (position >= getLength()) return 0
            base.position = position
            println("AsyncStream.check.read('$name':$position/${getLength()}): buffer(${buffer.size}), $offset, $len")
            return base.read(buffer, offset, len)
        }

        override suspend fun setLength(value: Long) {
            base.setLength(value)
        }

        override suspend fun write(position: Long, buffer: ByteArray, offset: Int, len: Int) {
            base.position = position
            base.write(buffer, offset, len)
        }
    }.toAsyncStream()
}

suspend fun AsyncStream.cached(name: String): AsyncStream {
    val base = this

    class CachedEntry(var position: Long = 0L, var data: ByteArray = ByteArray(0x10000), var dataSize: Int = 0) {
        val end: Long get() = position + dataSize
        val range get() = position until (position + dataSize)

        fun contains(position: Long, size: Int) = (position in range) && ((position + size - 1) in range)

        fun read(position: Long, buffer: ByteArray, offset: Int, len: Int): Int {
            if (len <= 0) return 0
            val roffset = (position - this.position).toInt()
            if (roffset !in 0 until dataSize) return 0
            val available = this.dataSize - roffset
            val alen = min(available, len)
            arraycopy(this.data, roffset, buffer, offset, alen)
            return alen
        }
    }

    var sslen = base.getLength()

    return object : AsyncStreamBase() {
        suspend override fun close() {
            base.close()
        }

        suspend override fun getLength(): Long = sslen

        val cacheEntry = CachedEntry()

        suspend override fun read(position: Long, buffer: ByteArray, offset: Int, len: Int): Int {
            if (position >= sslen) return 0

            if (!cacheEntry.contains(position, len)) {
                base.position = position
                println("[U]AsyncStream.check.read('$name':$position/$sslen): buffer(${buffer.size}), $offset, $len")
                val ret = base.read(cacheEntry.data, 0, cacheEntry.data.size)
                cacheEntry.position = position
                cacheEntry.dataSize = ret
            } else {
                println("[C]AsyncStream.check.read('$name':$position/$sslen): buffer(${buffer.size}), $offset, $len")
            }
            val ret = cacheEntry.read(position, buffer, offset, len)
            //println(buffer.hexString)
            println(" --> $ret")
            return ret
        }

        suspend override fun setLength(value: Long) {
            base.setLength(value)
            sslen = base.getLength()
        }

        suspend override fun write(position: Long, buffer: ByteArray, offset: Int, len: Int) {
            base.position = position
            base.write(buffer, offset, len)
        }
    }.toAsyncStream()
}