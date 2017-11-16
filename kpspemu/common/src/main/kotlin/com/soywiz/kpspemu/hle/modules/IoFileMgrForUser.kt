package com.soywiz.kpspemu.hle.modules

import com.soywiz.korio.async.*
import com.soywiz.korio.coroutine.getCoroutineContext
import com.soywiz.korio.error.invalidOp
import com.soywiz.korio.lang.UTF8
import com.soywiz.korio.lang.printStackTrace
import com.soywiz.korio.lang.toString
import com.soywiz.korio.util.toInt
import com.soywiz.korio.vfs.*
import com.soywiz.kpspemu.Emulator
import com.soywiz.kpspemu.cpu.CpuState
import com.soywiz.kpspemu.display
import com.soywiz.kpspemu.fileManager
import com.soywiz.kpspemu.hle.SceModule
import com.soywiz.kpspemu.hle.error.SceKernelErrors
import com.soywiz.kpspemu.hle.manager.*
import com.soywiz.kpspemu.mem.Ptr
import com.soywiz.kpspemu.mem.openSync
import com.soywiz.kpspemu.mem.readBytes
import com.soywiz.kpspemu.mem.writeBytes
import com.soywiz.kpspemu.util.AsyncPool2
import com.soywiz.kpspemu.util.PoolItem
import com.soywiz.kpspemu.util.Resetable
import com.soywiz.kpspemu.util.io.ISO2
import com.soywiz.kpspemu.util.write

@Suppress("UNUSED_PARAMETER")
class IoFileMgrForUser(emulator: Emulator) : SceModule(emulator, "IoFileMgrForUser", 0x40010011, "iofilemgr.prx", "sceIOFileManager") {
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

	private fun resolve(path: String?): VfsFile {
		logger.trace { "resolve:$path" }
		return _resolve(path!!)
	}

	suspend fun sceIoOpen(fileName: String?, flags: Int, mode: Int): Int {
		logger.warn { "WIP: sceIoOpen: $fileName, $flags, $mode" }
		try {
			val file = fileDescriptors.alloc()
			file.file = resolve(fileName)
			val flags2 = when {
				(flags and FileOpenFlags.Truncate) != 0 -> VfsOpenMode.CREATE_OR_TRUNCATE
				(flags and FileOpenFlags.Create) != 0 -> VfsOpenMode.CREATE
				(flags and FileOpenFlags.Write) != 0 -> VfsOpenMode.WRITE
				else -> VfsOpenMode.READ
			}
			file.stream = file.file.open(flags2)
			logger.warn { "WIP: sceIoOpen --> ${file.id}" }
			return file.id
		} catch (e: Throwable) {
			println("Error openingfile: $fileName : '${e.message}'")
			e.printStackTrace()
			//e.printStackTrace()
			return SceKernelErrors.ERROR_ERRNO_FILE_NOT_FOUND
		}
	}

	fun VfsStat.toSce() = SceIoStat(
		mode = 511 ,
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
		val stat = fstat.toSce()
		ptr.openSync().write(SceIoStat, stat)
		logger.warn { "sceIoGetstat --> 0" }
		return 0
	}

	suspend fun sceIoLseek32(fileId: Int, offset: Int, whence: Int): Int {
		return _sceIoLseek(fileId, offset.toLong(), whence).toInt()
	}

	suspend fun sceIoLseek(fileId: Int, offset: Long, whence: Int): Long {
		return _sceIoLseek(fileId, offset, whence)
	}

	suspend fun _sceIoLseek(fileId: Int, offset: Long, whence: Int): Long {
		logger.warn { "WIP: _sceIoLseek: $fileId, $offset, $whence" }
		val stream = fileDescriptors[fileId].stream
		stream.position = when (whence) {
			SeekType.Set -> offset
			SeekType.Cur -> stream.position + offset
			SeekType.End -> stream.size() + offset
			SeekType.Tell -> stream.position
			else -> invalidOp("Invalid sceIoLseek32")
		}
		return stream.position
	}


	suspend fun sceIoRead(fileId: Int, dst: Ptr, dstLen: Int): Int {
		logger.info { "WIP: sceIoRead: $fileId, $dst, $dstLen" }
		val stream = fileDescriptors[fileId].stream
		val out = ByteArray(dstLen)
		val read = stream.read(out, 0, dstLen)
		dst.writeBytes(out, 0, read)
		return read
	}

	suspend fun sceIoWrite(fileId: Int, ptr: Ptr, size: Int): Int {
		logger.info { "WIP: sceIoWrite: $fileId, $ptr, $size" }
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


	suspend fun sceIoDevctl(deviceName: String?, command: Int, inputPointer: Ptr, inputLength: Int, outputPointer: Ptr, outputLength: Int): Int {
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

		logger.error("WIP: sceIoDevctl: $deviceName, $command, $inputPointer, $inputLength, $outputPointer, $outputLength")

		return -1
	}

	suspend fun sceIoDopen(path: String?): Int {
		logger.error { "sceIoDopen:$path" }
		try {
			logger.error { "sceIoDopen:$path" }
			val dd = directoryDescriptors.alloc()
			dd.directory = resolve(path)
			dd.pos = 0
			dd.files = dd.directory.list().toList()
			return dd.id
		} catch (e: Throwable) {
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
		resolve(path).mkdir()
		return 0
	}

	class AsyncHandle(
		override val id: Int,
		var promise: Promise<Unit>? = null,
		var result: Long = 0L,
		var done: Boolean = false
	) : PoolItem {
		override fun reset() {
			promise = null
			result = 0L
			done = false
		}
	}
	val asyncPool = AsyncPool2<AsyncHandle>(initId = 1) { AsyncHandle(it) }

	suspend fun async(name: String, callback: suspend () -> Long): Int {
		logger.error { "starting async $name" }
		val res = asyncPool.alloc()
		res.done = false
		res.promise = spawn {
			res.result = callback()
			logger.error { "  async $name completed with result ${res.result}" }
			res.done = true
			getCoroutineContext().eventLoop.sleep(10000) // TODO: Delete after 10 seconds. Probably wrong and not time-related
			asyncPool.free(res)
		}
		logger.error { "  async $name ---> ${res.id}" }
		return res.id
	}

	suspend fun sceIoOpenAsync(filename: String?, flags: Int, mode: Int): Int {
		logger.error { "sceIoOpenAsync:$filename,$flags,$mode" }
		//val async = asyncPool.alloc()
		//async.promise = spawn {
		//	async.result = sceIoOpen(filename, flags, mode).toLong()
		//}
		//return async.id
		return async("sceIoOpenAsync") {
			val res = sceIoOpen(filename, flags, mode)
			logger.error { "sceIoOpenAsync --> $res" }
			res.toLong()
		}
	}

	suspend fun sceIoReadAsync(fileId: Int, outputPointer: Ptr, outputLength: Int): Int {
		logger.error { "sceIoReadAsync:$fileId,$outputPointer,$outputLength" }
		return async("sceIoReadAsync") {
			val res = sceIoRead(fileId, outputPointer, outputLength)
			logger.error { "sceIoReadAsync --> $res" }
			res.toLong()
		}
	}

	fun sceIoPollAsync(fd: Int, out: Ptr): Int {
		logger.error { "sceIoPollAsync:$fd,$out" }
		val res = asyncPool[fd] ?: return -1
		out.sdw(0, res.result)
		return if (res.done) 0 else 1
	}

	fun sceIoGetDevType(cpu: CpuState): Unit = UNIMPLEMENTED(0x08BD7374)
	fun sceIoWriteAsync(cpu: CpuState): Unit = UNIMPLEMENTED(0x0FACAB19)
	fun sceIoLseek32Async(cpu: CpuState): Unit = UNIMPLEMENTED(0x1B385D8F)
	fun sceIoWaitAsyncCB(cpu: CpuState): Unit = UNIMPLEMENTED(0x35DBD746)
	fun sceIoGetFdList(cpu: CpuState): Unit = UNIMPLEMENTED(0x5C2BE2CC)
	fun sceIoIoctl(cpu: CpuState): Unit = UNIMPLEMENTED(0x63632449)
	fun sceIoUnassign(cpu: CpuState): Unit = UNIMPLEMENTED(0x6D08A871)
	fun sceIoLseekAsync(cpu: CpuState): Unit = UNIMPLEMENTED(0x71B19E77)
	fun sceIoRename(cpu: CpuState): Unit = UNIMPLEMENTED(0x779103A0)
	fun sceIoSetAsyncCallback(cpu: CpuState): Unit = UNIMPLEMENTED(0xA12A0514)
	fun sceIoSync(cpu: CpuState): Unit = UNIMPLEMENTED(0xAB96437F)
	fun sceIoChangeAsyncPriority(cpu: CpuState): Unit = UNIMPLEMENTED(0xB293727F)
	fun sceIoAssign(cpu: CpuState): Unit = UNIMPLEMENTED(0xB2A628C1)
	fun sceIoChstat(cpu: CpuState): Unit = UNIMPLEMENTED(0xB8A740F4)
	fun sceIoGetAsyncStat(cpu: CpuState): Unit = UNIMPLEMENTED(0xCB05F8D6)
	fun sceIoWaitAsync(cpu: CpuState): Unit = UNIMPLEMENTED(0xE23EEC33)
	fun sceIoCancel(cpu: CpuState): Unit = UNIMPLEMENTED(0xE8BC6571)
	fun sceIoIoctlAsync(cpu: CpuState): Unit = UNIMPLEMENTED(0xE95A012B)
	fun sceIoRemove(cpu: CpuState): Unit = UNIMPLEMENTED(0xF27A9C51)
	fun sceIoCloseAsync(cpu: CpuState): Unit = UNIMPLEMENTED(0xFF5940B6)


	override fun registerModule() {
		// Devices
		registerFunctionSuspendInt("sceIoDevctl", 0x54F5FB11, since = 150) { sceIoDevctl(str, int, ptr, int, ptr, int) }

		// Files
		registerFunctionSuspendInt("sceIoOpen", 0x109F50BC, since = 150) { sceIoOpen(str, int, int) }
		registerFunctionSuspendInt("sceIoLseek32", 0x68963324, since = 150) { sceIoLseek32(int, int, int) }
		registerFunctionSuspendLong("sceIoLseek", 0x27EB27B8, since = 150) { sceIoLseek(int, long, int) }
		registerFunctionSuspendInt("sceIoWrite", 0x42EC03AC, since = 150) { sceIoWrite(int, ptr, int) }
		registerFunctionSuspendInt("sceIoRead", 0x6A638D83, since = 150) { sceIoRead(int, ptr, int) }
		registerFunctionSuspendInt("sceIoClose", 0x810C4BC3, since = 150) { sceIoClose(int) }
		registerFunctionSuspendInt("sceIoGetstat", 0xACE946E8, since = 150) { sceIoGetstat(str, ptr) }

		// Files Async
		registerFunctionSuspendInt("sceIoOpenAsync", 0x89AA9906, since = 150) { sceIoOpenAsync(str, int, int) }
		registerFunctionSuspendInt("sceIoReadAsync", 0xA0B5A7C2, since = 150) { sceIoReadAsync(int, ptr, int) }
		registerFunctionInt("sceIoPollAsync", 0x3251EA56, since = 150) { sceIoPollAsync(int, ptr) }


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
		registerFunctionRaw("sceIoWaitAsyncCB", 0x35DBD746, since = 150) { sceIoWaitAsyncCB(it) }
		registerFunctionRaw("sceIoGetFdList", 0x5C2BE2CC, since = 150) { sceIoGetFdList(it) }
		registerFunctionRaw("sceIoIoctl", 0x63632449, since = 150) { sceIoIoctl(it) }
		registerFunctionRaw("sceIoUnassign", 0x6D08A871, since = 150) { sceIoUnassign(it) }
		registerFunctionRaw("sceIoLseekAsync", 0x71B19E77, since = 150) { sceIoLseekAsync(it) }
		registerFunctionRaw("sceIoRename", 0x779103A0, since = 150) { sceIoRename(it) }
		registerFunctionRaw("sceIoSetAsyncCallback", 0xA12A0514, since = 150) { sceIoSetAsyncCallback(it) }
		registerFunctionRaw("sceIoSync", 0xAB96437F, since = 150) { sceIoSync(it) }
		registerFunctionRaw("sceIoChangeAsyncPriority", 0xB293727F, since = 150) { sceIoChangeAsyncPriority(it) }
		registerFunctionRaw("sceIoAssign", 0xB2A628C1, since = 150) { sceIoAssign(it) }
		registerFunctionRaw("sceIoChstat", 0xB8A740F4, since = 150) { sceIoChstat(it) }
		registerFunctionRaw("sceIoGetAsyncStat", 0xCB05F8D6, since = 150) { sceIoGetAsyncStat(it) }
		registerFunctionRaw("sceIoWaitAsync", 0xE23EEC33, since = 150) { sceIoWaitAsync(it) }
		registerFunctionRaw("sceIoCancel", 0xE8BC6571, since = 150) { sceIoCancel(it) }
		registerFunctionRaw("sceIoIoctlAsync", 0xE95A012B, since = 150) { sceIoIoctlAsync(it) }
		registerFunctionRaw("sceIoRemove", 0xF27A9C51, since = 150) { sceIoRemove(it) }
		registerFunctionRaw("sceIoCloseAsync", 0xFF5940B6, since = 150) { sceIoCloseAsync(it) }
	}
}
