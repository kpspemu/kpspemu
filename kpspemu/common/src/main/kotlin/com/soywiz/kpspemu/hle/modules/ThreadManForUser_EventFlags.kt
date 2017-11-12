package com.soywiz.kpspemu.hle.modules

import com.soywiz.kpspemu.cpu.CpuState
import com.soywiz.kpspemu.hle.error.SceKernelErrors
import com.soywiz.kpspemu.hle.error.sceKernelException
import com.soywiz.kpspemu.hle.manager.EventFlagWaitTypeSet
import com.soywiz.kpspemu.mem.Ptr

interface ThreadManForUser_EventFlags {
	fun ThreadManForUser.sceKernelCreateEventFlag(name: String?, attributes: Int, bitPattern: Int, optionsPtr: Ptr): Int {
		return eventFlags.alloc().apply {
			this.name = name ?: "eventFlag"
			this.attributes = attributes
			this.currentPattern = bitPattern
			this.optionsPtr = optionsPtr
		}.id
	}

	fun ThreadManForUser.sceKernelPollEventFlag(id: Int, bits: Int, waitType: Int, outBits: Ptr): Int {
		val eventFlag = eventFlags.tryGetById(id) ?: sceKernelException(SceKernelErrors.ERROR_KERNEL_NOT_FOUND_EVENT_FLAG)
		if ((waitType and EventFlagWaitTypeSet.MaskValidBits.inv()) != 0) return SceKernelErrors.ERROR_KERNEL_ILLEGAL_MODE
		if ((waitType and (EventFlagWaitTypeSet.Clear or EventFlagWaitTypeSet.ClearAll)) == (EventFlagWaitTypeSet.Clear or EventFlagWaitTypeSet.ClearAll)) {
			return SceKernelErrors.ERROR_KERNEL_ILLEGAL_MODE
		}
		if (bits == 0) return SceKernelErrors.ERROR_KERNEL_EVENT_FLAG_ILLEGAL_WAIT_PATTERN
		//if (EventFlag == null) return SceKernelErrors.ERROR_KERNEL_NOT_FOUND_EVENT_FLAG;

		val matched = eventFlag.poll(bits, waitType, outBits)

		return if (matched) 0 else SceKernelErrors.ERROR_KERNEL_EVENT_FLAG_POLL_FAILED
	}

	fun ThreadManForUser.sceKernelCancelEventFlag(cpu: CpuState): Unit = UNIMPLEMENTED(0xCD203292)
	fun ThreadManForUser.sceKernelReferEventFlagStatus(cpu: CpuState): Unit = UNIMPLEMENTED(0xA66B0120)
	fun ThreadManForUser.sceKernelSetEventFlag(cpu: CpuState): Unit = UNIMPLEMENTED(0x1FB15A32)
	fun ThreadManForUser.sceKernelWaitEventFlagCB(cpu: CpuState): Unit = UNIMPLEMENTED(0x328C546A)
	fun ThreadManForUser.sceKernelWaitEventFlag(cpu: CpuState): Unit = UNIMPLEMENTED(0x402FCF22)
	fun ThreadManForUser.sceKernelClearEventFlag(cpu: CpuState): Unit = UNIMPLEMENTED(0x812346E4)
	fun ThreadManForUser.sceKernelDeleteEventFlag(cpu: CpuState): Unit = UNIMPLEMENTED(0xEF9E4C70)

	fun ThreadManForUser.registerModuleEventFlags() {
		registerFunctionInt("sceKernelCreateEventFlag", 0x55C20A00, since = 150) { sceKernelCreateEventFlag(str, int, int, ptr) }
		registerFunctionInt("sceKernelPollEventFlag", 0x30FD48F0, since = 150) { sceKernelPollEventFlag(int, int, int, ptr) }

		registerFunctionRaw("sceKernelCancelEventFlag", 0xCD203292, since = 150) { sceKernelCancelEventFlag(it) }
		registerFunctionRaw("sceKernelReferEventFlagStatus", 0xA66B0120, since = 150) { sceKernelReferEventFlagStatus(it) }
		registerFunctionRaw("sceKernelSetEventFlag", 0x1FB15A32, since = 150) { sceKernelSetEventFlag(it) }
		registerFunctionRaw("sceKernelWaitEventFlagCB", 0x328C546A, since = 150) { sceKernelWaitEventFlagCB(it) }
		registerFunctionRaw("sceKernelWaitEventFlag", 0x402FCF22, since = 150) { sceKernelWaitEventFlag(it) }
		registerFunctionRaw("sceKernelClearEventFlag", 0x812346E4, since = 150) { sceKernelClearEventFlag(it) }
		registerFunctionRaw("sceKernelDeleteEventFlag", 0xEF9E4C70, since = 150) { sceKernelDeleteEventFlag(it) }
	}
}