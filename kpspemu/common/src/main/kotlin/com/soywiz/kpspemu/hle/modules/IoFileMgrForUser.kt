package com.soywiz.kpspemu.hle.modules


import com.soywiz.korio.lang.UTF8
import com.soywiz.korio.lang.toString
import com.soywiz.kpspemu.Emulator
import com.soywiz.kpspemu.cpu.CpuState
import com.soywiz.kpspemu.display
import com.soywiz.kpspemu.hle.SceModule
import com.soywiz.kpspemu.mem.Ptr
import com.soywiz.kpspemu.mem.readBytes
import com.soywiz.kpspemu.util.toInt


class IoFileMgrForUser(emulator: Emulator) : SceModule(emulator, "IoFileMgrForUser", 0x40010011, "iofilemgr.prx", "sceIOFileManager") {
	companion object {
		const val EMULATOR_DEVCTL__GET_HAS_DISPLAY = 0x00000001
		const val EMULATOR_DEVCTL__SEND_OUTPUT = 0x00000002
		const val EMULATOR_DEVCTL__IS_EMULATOR = 0x00000003
		const val EMULATOR_DEVCTL__SEND_CTRLDATA = 0x00000010
		const val EMULATOR_DEVCTL__EMIT_SCREENSHOT = 0x00000020
	}

	fun sceIoOpen(filename: String?, flags: Int, mode: Int): Int {
		println("WIP: sceIoOpen: $filename, $flags, $mode")
		return 0
	}

	fun sceIoWrite(fileId: Int, ptr: Ptr, size: Int): Int {
		println("WIP: sceIoWrite: $fileId, $ptr, $size")
		return 0
	}

	fun sceIoDevctl(deviceName: String?, command: Int, inputPointer: Ptr, inputLength: Int, outputPointer: Ptr, outputLength: Int): Int {
		when (deviceName) {
			"kemulator:", "emulator:" -> {
				when (command) {
					EMULATOR_DEVCTL__IS_EMULATOR -> return 0 // Yes, we are in an emulator!
					EMULATOR_DEVCTL__GET_HAS_DISPLAY -> { outputPointer.sw(0, display.exposeDisplay.toInt()); return 0; }
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

		println("WIP: sceIoDevctl: $deviceName, $command, $inputPointer, $inputLength, $outputPointer, $outputLength")

		return -1
	}

	fun sceIoMkdir(cpu: CpuState): Unit = UNIMPLEMENTED(0x06A70004)
	fun sceIoGetDevType(cpu: CpuState): Unit = UNIMPLEMENTED(0x08BD7374)
	fun sceIoWriteAsync(cpu: CpuState): Unit = UNIMPLEMENTED(0x0FACAB19)
	fun sceIoRmdir(cpu: CpuState): Unit = UNIMPLEMENTED(0x1117C65F)
	fun sceIoLseek32Async(cpu: CpuState): Unit = UNIMPLEMENTED(0x1B385D8F)
	fun sceIoLseek(cpu: CpuState): Unit = UNIMPLEMENTED(0x27EB27B8)
	fun sceIoPollAsync(cpu: CpuState): Unit = UNIMPLEMENTED(0x3251EA56)
	fun sceIoWaitAsyncCB(cpu: CpuState): Unit = UNIMPLEMENTED(0x35DBD746)
	fun sceIoChdir(cpu: CpuState): Unit = UNIMPLEMENTED(0x55F4717D)
	fun sceIoGetFdList(cpu: CpuState): Unit = UNIMPLEMENTED(0x5C2BE2CC)
	fun sceIoIoctl(cpu: CpuState): Unit = UNIMPLEMENTED(0x63632449)
	fun sceIoLseek32(cpu: CpuState): Unit = UNIMPLEMENTED(0x68963324)
	fun sceIoRead(cpu: CpuState): Unit = UNIMPLEMENTED(0x6A638D83)
	fun sceIoUnassign(cpu: CpuState): Unit = UNIMPLEMENTED(0x6D08A871)
	fun sceIoLseekAsync(cpu: CpuState): Unit = UNIMPLEMENTED(0x71B19E77)
	fun sceIoRename(cpu: CpuState): Unit = UNIMPLEMENTED(0x779103A0)
	fun sceIoClose(cpu: CpuState): Unit = UNIMPLEMENTED(0x810C4BC3)
	fun sceIoOpenAsync(cpu: CpuState): Unit = UNIMPLEMENTED(0x89AA9906)
	fun sceIoReadAsync(cpu: CpuState): Unit = UNIMPLEMENTED(0xA0B5A7C2)
	fun sceIoSetAsyncCallback(cpu: CpuState): Unit = UNIMPLEMENTED(0xA12A0514)
	fun sceIoSync(cpu: CpuState): Unit = UNIMPLEMENTED(0xAB96437F)
	fun sceIoGetstat(cpu: CpuState): Unit = UNIMPLEMENTED(0xACE946E8)
	fun sceIoChangeAsyncPriority(cpu: CpuState): Unit = UNIMPLEMENTED(0xB293727F)
	fun sceIoDopen(cpu: CpuState): Unit = UNIMPLEMENTED(0xB29DDF9C)
	fun sceIoAssign(cpu: CpuState): Unit = UNIMPLEMENTED(0xB2A628C1)
	fun sceIoChstat(cpu: CpuState): Unit = UNIMPLEMENTED(0xB8A740F4)
	fun sceIoGetAsyncStat(cpu: CpuState): Unit = UNIMPLEMENTED(0xCB05F8D6)
	fun sceIoWaitAsync(cpu: CpuState): Unit = UNIMPLEMENTED(0xE23EEC33)
	fun sceIoDread(cpu: CpuState): Unit = UNIMPLEMENTED(0xE3EB004C)
	fun sceIoCancel(cpu: CpuState): Unit = UNIMPLEMENTED(0xE8BC6571)
	fun sceIoIoctlAsync(cpu: CpuState): Unit = UNIMPLEMENTED(0xE95A012B)
	fun sceIoDclose(cpu: CpuState): Unit = UNIMPLEMENTED(0xEB092469)
	fun sceIoRemove(cpu: CpuState): Unit = UNIMPLEMENTED(0xF27A9C51)
	fun sceIoCloseAsync(cpu: CpuState): Unit = UNIMPLEMENTED(0xFF5940B6)


	override fun registerModule() {
		registerFunctionInt("sceIoOpen", 0x109F50BC, since = 150) { sceIoOpen(str, int, int) }
		registerFunctionInt("sceIoWrite", 0x42EC03AC, since = 150) { sceIoWrite(int, ptr, int) }
		registerFunctionInt("sceIoDevctl", 0x54F5FB11, since = 150) { sceIoDevctl(str, int, ptr, int, ptr, int) }

		registerFunctionRaw("sceIoMkdir", 0x06A70004, since = 150) { sceIoMkdir(it) }
		registerFunctionRaw("sceIoGetDevType", 0x08BD7374, since = 150) { sceIoGetDevType(it) }
		registerFunctionRaw("sceIoWriteAsync", 0x0FACAB19, since = 150) { sceIoWriteAsync(it) }
		registerFunctionRaw("sceIoRmdir", 0x1117C65F, since = 150) { sceIoRmdir(it) }
		registerFunctionRaw("sceIoLseek32Async", 0x1B385D8F, since = 150) { sceIoLseek32Async(it) }
		registerFunctionRaw("sceIoLseek", 0x27EB27B8, since = 150) { sceIoLseek(it) }
		registerFunctionRaw("sceIoPollAsync", 0x3251EA56, since = 150) { sceIoPollAsync(it) }
		registerFunctionRaw("sceIoWaitAsyncCB", 0x35DBD746, since = 150) { sceIoWaitAsyncCB(it) }
		registerFunctionRaw("sceIoChdir", 0x55F4717D, since = 150) { sceIoChdir(it) }
		registerFunctionRaw("sceIoGetFdList", 0x5C2BE2CC, since = 150) { sceIoGetFdList(it) }
		registerFunctionRaw("sceIoIoctl", 0x63632449, since = 150) { sceIoIoctl(it) }
		registerFunctionRaw("sceIoLseek32", 0x68963324, since = 150) { sceIoLseek32(it) }
		registerFunctionRaw("sceIoRead", 0x6A638D83, since = 150) { sceIoRead(it) }
		registerFunctionRaw("sceIoUnassign", 0x6D08A871, since = 150) { sceIoUnassign(it) }
		registerFunctionRaw("sceIoLseekAsync", 0x71B19E77, since = 150) { sceIoLseekAsync(it) }
		registerFunctionRaw("sceIoRename", 0x779103A0, since = 150) { sceIoRename(it) }
		registerFunctionRaw("sceIoClose", 0x810C4BC3, since = 150) { sceIoClose(it) }
		registerFunctionRaw("sceIoOpenAsync", 0x89AA9906, since = 150) { sceIoOpenAsync(it) }
		registerFunctionRaw("sceIoReadAsync", 0xA0B5A7C2, since = 150) { sceIoReadAsync(it) }
		registerFunctionRaw("sceIoSetAsyncCallback", 0xA12A0514, since = 150) { sceIoSetAsyncCallback(it) }
		registerFunctionRaw("sceIoSync", 0xAB96437F, since = 150) { sceIoSync(it) }
		registerFunctionRaw("sceIoGetstat", 0xACE946E8, since = 150) { sceIoGetstat(it) }
		registerFunctionRaw("sceIoChangeAsyncPriority", 0xB293727F, since = 150) { sceIoChangeAsyncPriority(it) }
		registerFunctionRaw("sceIoDopen", 0xB29DDF9C, since = 150) { sceIoDopen(it) }
		registerFunctionRaw("sceIoAssign", 0xB2A628C1, since = 150) { sceIoAssign(it) }
		registerFunctionRaw("sceIoChstat", 0xB8A740F4, since = 150) { sceIoChstat(it) }
		registerFunctionRaw("sceIoGetAsyncStat", 0xCB05F8D6, since = 150) { sceIoGetAsyncStat(it) }
		registerFunctionRaw("sceIoWaitAsync", 0xE23EEC33, since = 150) { sceIoWaitAsync(it) }
		registerFunctionRaw("sceIoDread", 0xE3EB004C, since = 150) { sceIoDread(it) }
		registerFunctionRaw("sceIoCancel", 0xE8BC6571, since = 150) { sceIoCancel(it) }
		registerFunctionRaw("sceIoIoctlAsync", 0xE95A012B, since = 150) { sceIoIoctlAsync(it) }
		registerFunctionRaw("sceIoDclose", 0xEB092469, since = 150) { sceIoDclose(it) }
		registerFunctionRaw("sceIoRemove", 0xF27A9C51, since = 150) { sceIoRemove(it) }
		registerFunctionRaw("sceIoCloseAsync", 0xFF5940B6, since = 150) { sceIoCloseAsync(it) }
	}
}
