package com.soywiz.kpspemu.hle.modules

import com.soywiz.korio.error.invalidOp
import com.soywiz.kpspemu.Emulator
import com.soywiz.kpspemu.callbackManager
import com.soywiz.kpspemu.cpu.CpuState
import com.soywiz.kpspemu.cpu.GP
import com.soywiz.kpspemu.hle.SceModule
import com.soywiz.kpspemu.hle.manager.PspThread
import com.soywiz.kpspemu.hle.manager.WaitObject
import com.soywiz.kpspemu.mem.Ptr
import com.soywiz.kpspemu.mem.isNotNull
import com.soywiz.kpspemu.mem.readBytes
import com.soywiz.kpspemu.rtc
import com.soywiz.kpspemu.threadManager
import com.soywiz.kpspemu.util.ResourceItem
import com.soywiz.kpspemu.util.ResourceList

@Suppress("UNUSED_PARAMETER")
class ThreadManForUser(emulator: Emulator) : SceModule(emulator, "ThreadManForUser", 0x40010011, "threadman.prx", "sceThreadManager") {
	fun sceKernelCreateThread(name: String?, entryPoint: Int, initPriority: Int, stackSize: Int, attributes: Int, optionPtr: Ptr): Int {
		val thread = threadManager.create(name ?: "unknown", entryPoint, initPriority, stackSize, attributes, optionPtr)
		//println("sceKernelCreateThread: ${thread.id}")
		return thread.id
	}

	fun sceKernelStartThread(currentThread: PspThread, threadId: Int, userDataLength: Int, userDataPtr: Ptr): Int {
		//println("sceKernelStartThread: $threadId")
		val thread = threadManager.resourcesById[threadId] ?: invalidOp("Can't find thread $threadId")
		if (userDataPtr.isNotNull) {
			val localUserDataPtr = thread.putDataInStack(userDataPtr.readBytes(userDataLength))
			thread.state.r4 = userDataLength
			thread.state.r5 = localUserDataPtr.addr
		} else {
			thread.state.r4 = 0
			thread.state.r5 = 0
		}
		thread.state.GP = currentThread.state.GP
		thread.start()
		return 0
	}

	fun sceKernelSleepThreadCB(currentThread: PspThread): Int {
		currentThread.suspend(WaitObject.SLEEP, cb = true)
		return 0
	}

	fun sceKernelGetThreadCurrentPriority(thread: PspThread): Int = thread.priority

	fun sceKernelGetSystemTimeWide(): Long = rtc.getTimeInMicroseconds()
	fun sceKernelGetSystemTimeLow(): Int = rtc.getTimeInMicroseconds().toInt()

	fun sceKernelCreateCallback(name: String?, func: Ptr, arg: Int): Int {
		val callback = callbackManager.create(name ?: "callback", func, arg)
		return callback.id
	}

	val eventFlags = ResourceList("EventFlag") { PspEventFlag(it) }

	fun sceKernelCreateEventFlag(name: String?, attributes: Int, bitPattern: Int, optionsPtr: Ptr): Int {
		return eventFlags.alloc().apply {
			this.name = name ?: "eventFlag"
			this.attributes = attributes
			this.bitPattern = bitPattern
			this.optionsPtr = optionsPtr
		}.id
	}

	fun _sceKernelDelayThread(thread: PspThread, microseconds: Int, cb: Boolean): Int {
		thread.suspend(WaitObject.TIME(rtc.getTimeInMicroseconds() + microseconds), cb = cb)
		return 0
	}

	fun sceKernelDelayThreadCB(thread: PspThread, microseconds: Int): Int = _sceKernelDelayThread(thread, microseconds, cb = true)
	fun sceKernelDelayThread(thread: PspThread, microseconds: Int): Int = _sceKernelDelayThread(thread, microseconds, cb = false)

	fun sceKernelGetVTimerTime(cpu: CpuState): Unit = UNIMPLEMENTED(0x034A921F)
	fun sceKernelRegisterThreadEventHandler(cpu: CpuState): Unit = UNIMPLEMENTED(0x0C106E53)
	fun sceKernelPollMbx(cpu: CpuState): Unit = UNIMPLEMENTED(0x0D81716A)
	fun sceKernelTryLockMutex(cpu: CpuState): Unit = UNIMPLEMENTED(0x0DDCD2C9)
	fun _sceKernelReturnFromTimerHandler(cpu: CpuState): Unit = UNIMPLEMENTED(0x0E927AED)
	fun sceKernelUSec2SysClock(cpu: CpuState): Unit = UNIMPLEMENTED(0x110DEC9A)
	fun sceKernelDelaySysClockThreadCB(cpu: CpuState): Unit = UNIMPLEMENTED(0x1181E963)
	fun sceKernelReferThreadStatus(cpu: CpuState): Unit = UNIMPLEMENTED(0x17C1684E)
	fun sceKernelReceiveMbx(cpu: CpuState): Unit = UNIMPLEMENTED(0x18260574)
	fun sceKernelCreateLwMutex(cpu: CpuState): Unit = UNIMPLEMENTED(0x19CFF145)
	fun sceKernelDonateWakeupThread(cpu: CpuState): Unit = UNIMPLEMENTED(0x1AF94D03)
	fun sceKernelCancelVpl(cpu: CpuState): Unit = UNIMPLEMENTED(0x1D371B8A)
	fun sceKernelSetEventFlag(cpu: CpuState): Unit = UNIMPLEMENTED(0x1FB15A32)
	fun sceKernelCreateVTimer(cpu: CpuState): Unit = UNIMPLEMENTED(0x20FFF560)
	fun sceKernelWaitThreadEnd(cpu: CpuState): Unit = UNIMPLEMENTED(0x278C0DF5)
	fun sceKernelResumeDispatchThread(cpu: CpuState): Unit = UNIMPLEMENTED(0x27E22EC2)
	fun sceKernelDeleteSema(cpu: CpuState): Unit = UNIMPLEMENTED(0x28B6489C)
	fun sceKernelGetThreadId(cpu: CpuState): Unit = UNIMPLEMENTED(0x293B45B8)
	fun sceKernelGetCallbackCount(cpu: CpuState): Unit = UNIMPLEMENTED(0x2A3D44FF)
	fun sceKernelReleaseWaitThread(cpu: CpuState): Unit = UNIMPLEMENTED(0x2C34E053)
	fun sceKernelPollEventFlag(cpu: CpuState): Unit = UNIMPLEMENTED(0x30FD48F0)
	fun ThreadManForUser_31327F19(cpu: CpuState): Unit = UNIMPLEMENTED(0x31327F19)
	fun sceKernelWaitEventFlagCB(cpu: CpuState): Unit = UNIMPLEMENTED(0x328C546A)
	fun sceKernelDeleteVTimer(cpu: CpuState): Unit = UNIMPLEMENTED(0x328F9E52)
	fun sceKernelReferMsgPipeStatus(cpu: CpuState): Unit = UNIMPLEMENTED(0x33BE4024)
	fun sceKernelCancelMsgPipe(cpu: CpuState): Unit = UNIMPLEMENTED(0x349B864D)
	fun sceKernelCheckCallback(cpu: CpuState): Unit = UNIMPLEMENTED(0x349D6D6C)
	fun sceKernelReferThreadEventHandlerStatus(cpu: CpuState): Unit = UNIMPLEMENTED(0x369EEB6B)
	fun sceKernelTerminateDeleteThread(cpu: CpuState): Unit = UNIMPLEMENTED(0x383F7BCC)
	fun sceKernelReferVplStatus(cpu: CpuState): Unit = UNIMPLEMENTED(0x39810265)
	fun sceKernelSuspendDispatchThread(cpu: CpuState): Unit = UNIMPLEMENTED(0x3AD58B8C)
	fun sceKernelGetThreadExitStatus(cpu: CpuState): Unit = UNIMPLEMENTED(0x3B183E26)
	fun sceKernelSignalSema(cpu: CpuState): Unit = UNIMPLEMENTED(0x3F53E640)
	fun sceKernelWaitEventFlag(cpu: CpuState): Unit = UNIMPLEMENTED(0x402FCF22)

	fun sceKernelReferLwMutexStatusByID(cpu: CpuState): Unit = UNIMPLEMENTED(0x4C145944)
	fun sceKernelWaitSema(cpu: CpuState): Unit = UNIMPLEMENTED(0x4E3A1105)
	fun sceKernelGetThreadStackFreeSize(cpu: CpuState): Unit = UNIMPLEMENTED(0x52089CA1)
	fun _sceKernelExitThread(cpu: CpuState): Unit = UNIMPLEMENTED(0x532A522E)
	fun sceKernelSetVTimerHandlerWide(cpu: CpuState): Unit = UNIMPLEMENTED(0x53B00E9A)
	fun sceKernelSetVTimerTime(cpu: CpuState): Unit = UNIMPLEMENTED(0x542AD630)
	fun sceKernelCreateVpl(cpu: CpuState): Unit = UNIMPLEMENTED(0x56C039B5)
	fun sceKernelGetThreadmanIdType(cpu: CpuState): Unit = UNIMPLEMENTED(0x57CF62DD)
	fun sceKernelPollSema(cpu: CpuState): Unit = UNIMPLEMENTED(0x58B1F937)
	fun sceKernelLockMutexCB(cpu: CpuState): Unit = UNIMPLEMENTED(0x5BF4DD27)
	fun sceKernelReferVTimerStatus(cpu: CpuState): Unit = UNIMPLEMENTED(0x5F32BEAA)
	fun sceKernelDeleteLwMutex(cpu: CpuState): Unit = UNIMPLEMENTED(0x60107536)
	fun sceKernelTerminateThread(cpu: CpuState): Unit = UNIMPLEMENTED(0x616403BA)
	fun sceKernelTryAllocateFpl(cpu: CpuState): Unit = UNIMPLEMENTED(0x623AE665)
	fun sceKernelReferSystemStatus(cpu: CpuState): Unit = UNIMPLEMENTED(0x627E6F3A)
	fun sceKernelReferThreadProfiler(cpu: CpuState): Unit = UNIMPLEMENTED(0x64D4540E)
	fun sceKernelSetAlarm(cpu: CpuState): Unit = UNIMPLEMENTED(0x6652B8CA)
	fun sceKernelUnlockMutex(cpu: CpuState): Unit = UNIMPLEMENTED(0x6B30100F)
	fun sceKernelWaitSemaCB(cpu: CpuState): Unit = UNIMPLEMENTED(0x6D212BAC)
	fun _sceKernelReturnFromCallback(cpu: CpuState): Unit = UNIMPLEMENTED(0x6E9EA350)
	fun ThreadManForUser_71040D5C(cpu: CpuState): Unit = UNIMPLEMENTED(0x71040D5C)
	fun sceKernelChangeThreadPriority(cpu: CpuState): Unit = UNIMPLEMENTED(0x71BC9871)
	fun sceKernelReleaseThreadEventHandler(cpu: CpuState): Unit = UNIMPLEMENTED(0x72F3C145)
	fun sceKernelReferCallbackStatus(cpu: CpuState): Unit = UNIMPLEMENTED(0x730ED8BC)
	fun sceKernelReceiveMsgPipe(cpu: CpuState): Unit = UNIMPLEMENTED(0x74829B76)
	fun sceKernelResumeThread(cpu: CpuState): Unit = UNIMPLEMENTED(0x75156E8F)
	fun sceKernelCreateMsgPipe(cpu: CpuState): Unit = UNIMPLEMENTED(0x7C0DC2A0)
	fun sceKernelSendMsgPipeCB(cpu: CpuState): Unit = UNIMPLEMENTED(0x7C41F2C2)
	fun ThreadManForUser_7CFF8CF3(cpu: CpuState): Unit = UNIMPLEMENTED(0x7CFF8CF3)
	fun sceKernelCancelAlarm(cpu: CpuState): Unit = UNIMPLEMENTED(0x7E65B999)
	fun sceKernelExitDeleteThread(cpu: CpuState): Unit = UNIMPLEMENTED(0x809CE29B)
	fun sceKernelClearEventFlag(cpu: CpuState): Unit = UNIMPLEMENTED(0x812346E4)
	fun sceKernelCreateMbx(cpu: CpuState): Unit = UNIMPLEMENTED(0x8125221D)
	fun sceKernelReferGlobalProfiler(cpu: CpuState): Unit = UNIMPLEMENTED(0x8218B4DD)
	fun sceKernelWaitThreadEndCB(cpu: CpuState): Unit = UNIMPLEMENTED(0x840E8133)
	fun sceKernelDeleteMbx(cpu: CpuState): Unit = UNIMPLEMENTED(0x86255ADA)
	fun ThreadManForUser_8672E3D0(cpu: CpuState): Unit = UNIMPLEMENTED(0x8672E3D0)
	fun sceKernelSendMsgPipe(cpu: CpuState): Unit = UNIMPLEMENTED(0x876DBFAD)
	fun sceKernelCancelReceiveMbx(cpu: CpuState): Unit = UNIMPLEMENTED(0x87D4DD36)
	fun sceKernelCancelMutex(cpu: CpuState): Unit = UNIMPLEMENTED(0x87D9223C)
	fun sceKernelTrySendMsgPipe(cpu: CpuState): Unit = UNIMPLEMENTED(0x884C9F90)
	fun sceKernelDeleteVpl(cpu: CpuState): Unit = UNIMPLEMENTED(0x89B3D48C)
	fun sceKernelCancelSema(cpu: CpuState): Unit = UNIMPLEMENTED(0x8FFDF9A2)
	fun sceKernelRotateThreadReadyQueue(cpu: CpuState): Unit = UNIMPLEMENTED(0x912354A7)
	fun sceKernelGetThreadmanIdList(cpu: CpuState): Unit = UNIMPLEMENTED(0x94416130)
	fun sceKernelSuspendThread(cpu: CpuState): Unit = UNIMPLEMENTED(0x9944F31F)
	fun sceKernelSleepThread(cpu: CpuState): Unit = UNIMPLEMENTED(0x9ACE131E)
	fun sceKernelDeleteThread(cpu: CpuState): Unit = UNIMPLEMENTED(0x9FA03CD3)
	fun sceKernelReferEventFlagStatus(cpu: CpuState): Unit = UNIMPLEMENTED(0xA66B0120)
	fun sceKernelCancelFpl(cpu: CpuState): Unit = UNIMPLEMENTED(0xA8AA591F)
	fun sceKernelReferMbxStatus(cpu: CpuState): Unit = UNIMPLEMENTED(0xA8E8C846)
	fun sceKernelReferMutexStatus(cpu: CpuState): Unit = UNIMPLEMENTED(0xA9C2CB9A)
	fun sceKernelExitThread(cpu: CpuState): Unit = UNIMPLEMENTED(0xAA73C935)
	fun sceKernelTryAllocateVpl(cpu: CpuState): Unit = UNIMPLEMENTED(0xAF36D708)
	fun sceKernelLockMutex(cpu: CpuState): Unit = UNIMPLEMENTED(0xB011B11F)
	fun sceKernelSetSysClockAlarm(cpu: CpuState): Unit = UNIMPLEMENTED(0xB2C25152)
	fun sceKernelGetVTimerBase(cpu: CpuState): Unit = UNIMPLEMENTED(0xB3A59970)
	fun sceKernelFreeVpl(cpu: CpuState): Unit = UNIMPLEMENTED(0xB736E9FF)
	fun sceKernelGetVTimerBaseWide(cpu: CpuState): Unit = UNIMPLEMENTED(0xB7C18B77)
	fun sceKernelCreateMutex(cpu: CpuState): Unit = UNIMPLEMENTED(0xB7D098C6)
	fun sceKernelCancelCallback(cpu: CpuState): Unit = UNIMPLEMENTED(0xBA4051D6)
	fun sceKernelSysClock2USec(cpu: CpuState): Unit = UNIMPLEMENTED(0xBA6B92E2)
	fun sceKernelReferSemaStatus(cpu: CpuState): Unit = UNIMPLEMENTED(0xBC6FEBC5)
	fun sceKernelDelaySysClockThread(cpu: CpuState): Unit = UNIMPLEMENTED(0xBD123D9E)
	fun sceKernelAllocateVpl(cpu: CpuState): Unit = UNIMPLEMENTED(0xBED27435)
	fun ThreadManForUser_BEED3A47(cpu: CpuState): Unit = UNIMPLEMENTED(0xBEED3A47)
	fun sceKernelCreateFpl(cpu: CpuState): Unit = UNIMPLEMENTED(0xC07BB470)
	fun sceKernelGetVTimerTimeWide(cpu: CpuState): Unit = UNIMPLEMENTED(0xC0B3FFD2)
	fun sceKernelNotifyCallback(cpu: CpuState): Unit = UNIMPLEMENTED(0xC11BA8C4)
	fun sceKernelStartVTimer(cpu: CpuState): Unit = UNIMPLEMENTED(0xC68D9437)
	fun sceKernelUSec2SysClockWide(cpu: CpuState): Unit = UNIMPLEMENTED(0xC8CD158C)
	fun sceKernelCancelEventFlag(cpu: CpuState): Unit = UNIMPLEMENTED(0xCD203292)
	fun sceKernelStopVTimer(cpu: CpuState): Unit = UNIMPLEMENTED(0xD0AEEE87)
	fun sceKernelCheckThreadStack(cpu: CpuState): Unit = UNIMPLEMENTED(0xD13BDE95)
	fun sceKernelCancelVTimerHandler(cpu: CpuState): Unit = UNIMPLEMENTED(0xD2D615EF)
	fun sceKernelWakeupThread(cpu: CpuState): Unit = UNIMPLEMENTED(0xD59EAD2F)
	fun sceKernelCreateSema(cpu: CpuState): Unit = UNIMPLEMENTED(0xD6DA4BA1)
	fun sceKernelReferFplStatus(cpu: CpuState): Unit = UNIMPLEMENTED(0xD8199E4C)
	fun sceKernelSetVTimerHandler(cpu: CpuState): Unit = UNIMPLEMENTED(0xD8B299AE)
	fun sceKernelAllocateFpl(cpu: CpuState): Unit = UNIMPLEMENTED(0xD979E9BF)
	fun sceKernelReferAlarmStatus(cpu: CpuState): Unit = UNIMPLEMENTED(0xDAA3F564)
	fun sceKernelGetSystemTime(cpu: CpuState): Unit = UNIMPLEMENTED(0xDB738F35)
	fun sceKernelTryReceiveMsgPipe(cpu: CpuState): Unit = UNIMPLEMENTED(0xDF52098F)
	fun sceKernelSysClock2USecWide(cpu: CpuState): Unit = UNIMPLEMENTED(0xE1619D7C)
	fun sceKernelAllocateFplCB(cpu: CpuState): Unit = UNIMPLEMENTED(0xE7282CB6)
	fun sceKernelSendMbx(cpu: CpuState): Unit = UNIMPLEMENTED(0xE9B3061E)
	fun sceKernelChangeCurrentThreadAttr(cpu: CpuState): Unit = UNIMPLEMENTED(0xEA748E31)
	fun sceKernelAllocateVplCB(cpu: CpuState): Unit = UNIMPLEMENTED(0xEC0A693F)
	fun sceKernelDeleteFpl(cpu: CpuState): Unit = UNIMPLEMENTED(0xED1410E0)
	fun sceKernelDeleteCallback(cpu: CpuState): Unit = UNIMPLEMENTED(0xEDBA5844)
	fun sceKernelDeleteEventFlag(cpu: CpuState): Unit = UNIMPLEMENTED(0xEF9E4C70)
	fun sceKernelDeleteMsgPipe(cpu: CpuState): Unit = UNIMPLEMENTED(0xF0B7DA1C)
	fun sceKernelReceiveMbxCB(cpu: CpuState): Unit = UNIMPLEMENTED(0xF3986382)
	fun sceKernelFreeFpl(cpu: CpuState): Unit = UNIMPLEMENTED(0xF6414A71)
	fun sceKernelDeleteMutex(cpu: CpuState): Unit = UNIMPLEMENTED(0xF8170FBE)
	fun sceKernelSetVTimerTimeWide(cpu: CpuState): Unit = UNIMPLEMENTED(0xFB6425C3)
	fun sceKernelReceiveMsgPipeCB(cpu: CpuState): Unit = UNIMPLEMENTED(0xFBFA697D)
	fun sceKernelCancelWakeupThread(cpu: CpuState): Unit = UNIMPLEMENTED(0xFCCFAD26)
	fun sceKernelReferThreadRunStatus(cpu: CpuState): Unit = UNIMPLEMENTED(0xFFC36A14)

	override fun registerModule() {
		// Time
		registerFunctionLong("sceKernelGetSystemTimeWide", 0x82BC5777, since = 150) { sceKernelGetSystemTimeWide() }
		registerFunctionInt("sceKernelGetSystemTimeLow", 0x369ED59D, since = 150) { sceKernelGetSystemTimeLow() }

		// Thread
		registerFunctionInt("sceKernelCreateThread", 0x446D8DE6, since = 150) { sceKernelCreateThread(str, int, int, int, int, ptr) }
		registerFunctionInt("sceKernelStartThread", 0xF475845D, since = 150) { sceKernelStartThread(thread, int, int, ptr) }
		registerFunctionInt("sceKernelSleepThreadCB", 0x82826F70, since = 150) { sceKernelSleepThreadCB(thread) }
		registerFunctionInt("sceKernelGetThreadCurrentPriority", 0x94AA61EE, since = 150) { sceKernelGetThreadCurrentPriority(thread) }
		registerFunctionInt("sceKernelDelayThreadCB", 0x68DA9E36, since = 150) { sceKernelDelayThreadCB(thread, int) }
		registerFunctionInt("sceKernelDelayThread", 0xCEADEB47, since = 150) { sceKernelDelayThread(thread, int) }

		// Callbacks
		registerFunctionInt("sceKernelCreateCallback", 0xE81CAF8F, since = 150) { sceKernelCreateCallback(str, ptr, int) }

		// EventFlags
		registerFunctionInt("sceKernelCreateEventFlag", 0x55C20A00, since = 150) { sceKernelCreateEventFlag(str, int, int, ptr) }

		registerFunctionRaw("sceKernelGetVTimerTime", 0x034A921F, since = 150) { sceKernelGetVTimerTime(it) }
		registerFunctionRaw("sceKernelRegisterThreadEventHandler", 0x0C106E53, since = 150) { sceKernelRegisterThreadEventHandler(it) }
		registerFunctionRaw("sceKernelPollMbx", 0x0D81716A, since = 150) { sceKernelPollMbx(it) }
		registerFunctionRaw("sceKernelTryLockMutex", 0x0DDCD2C9, since = 150) { sceKernelTryLockMutex(it) }
		registerFunctionRaw("_sceKernelReturnFromTimerHandler", 0x0E927AED, since = 150) { _sceKernelReturnFromTimerHandler(it) }
		registerFunctionRaw("sceKernelUSec2SysClock", 0x110DEC9A, since = 150) { sceKernelUSec2SysClock(it) }
		registerFunctionRaw("sceKernelDelaySysClockThreadCB", 0x1181E963, since = 150) { sceKernelDelaySysClockThreadCB(it) }
		registerFunctionRaw("sceKernelReferThreadStatus", 0x17C1684E, since = 150) { sceKernelReferThreadStatus(it) }
		registerFunctionRaw("sceKernelReceiveMbx", 0x18260574, since = 150) { sceKernelReceiveMbx(it) }
		registerFunctionRaw("sceKernelCreateLwMutex", 0x19CFF145, since = 150) { sceKernelCreateLwMutex(it) }
		registerFunctionRaw("sceKernelDonateWakeupThread", 0x1AF94D03, since = 150) { sceKernelDonateWakeupThread(it) }
		registerFunctionRaw("sceKernelCancelVpl", 0x1D371B8A, since = 150) { sceKernelCancelVpl(it) }
		registerFunctionRaw("sceKernelSetEventFlag", 0x1FB15A32, since = 150) { sceKernelSetEventFlag(it) }
		registerFunctionRaw("sceKernelCreateVTimer", 0x20FFF560, since = 150) { sceKernelCreateVTimer(it) }
		registerFunctionRaw("sceKernelWaitThreadEnd", 0x278C0DF5, since = 150) { sceKernelWaitThreadEnd(it) }
		registerFunctionRaw("sceKernelResumeDispatchThread", 0x27E22EC2, since = 150) { sceKernelResumeDispatchThread(it) }
		registerFunctionRaw("sceKernelDeleteSema", 0x28B6489C, since = 150) { sceKernelDeleteSema(it) }
		registerFunctionRaw("sceKernelGetThreadId", 0x293B45B8, since = 150) { sceKernelGetThreadId(it) }
		registerFunctionRaw("sceKernelGetCallbackCount", 0x2A3D44FF, since = 150) { sceKernelGetCallbackCount(it) }
		registerFunctionRaw("sceKernelReleaseWaitThread", 0x2C34E053, since = 150) { sceKernelReleaseWaitThread(it) }
		registerFunctionRaw("sceKernelPollEventFlag", 0x30FD48F0, since = 150) { sceKernelPollEventFlag(it) }
		registerFunctionRaw("ThreadManForUser_31327F19", 0x31327F19, since = 150) { ThreadManForUser_31327F19(it) }
		registerFunctionRaw("sceKernelWaitEventFlagCB", 0x328C546A, since = 150) { sceKernelWaitEventFlagCB(it) }
		registerFunctionRaw("sceKernelDeleteVTimer", 0x328F9E52, since = 150) { sceKernelDeleteVTimer(it) }
		registerFunctionRaw("sceKernelReferMsgPipeStatus", 0x33BE4024, since = 150) { sceKernelReferMsgPipeStatus(it) }
		registerFunctionRaw("sceKernelCancelMsgPipe", 0x349B864D, since = 150) { sceKernelCancelMsgPipe(it) }
		registerFunctionRaw("sceKernelCheckCallback", 0x349D6D6C, since = 150) { sceKernelCheckCallback(it) }
		registerFunctionRaw("sceKernelReferThreadEventHandlerStatus", 0x369EEB6B, since = 150) { sceKernelReferThreadEventHandlerStatus(it) }
		registerFunctionRaw("sceKernelTerminateDeleteThread", 0x383F7BCC, since = 150) { sceKernelTerminateDeleteThread(it) }
		registerFunctionRaw("sceKernelReferVplStatus", 0x39810265, since = 150) { sceKernelReferVplStatus(it) }
		registerFunctionRaw("sceKernelSuspendDispatchThread", 0x3AD58B8C, since = 150) { sceKernelSuspendDispatchThread(it) }
		registerFunctionRaw("sceKernelGetThreadExitStatus", 0x3B183E26, since = 150) { sceKernelGetThreadExitStatus(it) }
		registerFunctionRaw("sceKernelSignalSema", 0x3F53E640, since = 150) { sceKernelSignalSema(it) }
		registerFunctionRaw("sceKernelWaitEventFlag", 0x402FCF22, since = 150) { sceKernelWaitEventFlag(it) }
		registerFunctionRaw("sceKernelReferLwMutexStatusByID", 0x4C145944, since = 150) { sceKernelReferLwMutexStatusByID(it) }
		registerFunctionRaw("sceKernelWaitSema", 0x4E3A1105, since = 150) { sceKernelWaitSema(it) }
		registerFunctionRaw("sceKernelGetThreadStackFreeSize", 0x52089CA1, since = 150) { sceKernelGetThreadStackFreeSize(it) }
		registerFunctionRaw("_sceKernelExitThread", 0x532A522E, since = 150) { _sceKernelExitThread(it) }
		registerFunctionRaw("sceKernelSetVTimerHandlerWide", 0x53B00E9A, since = 150) { sceKernelSetVTimerHandlerWide(it) }
		registerFunctionRaw("sceKernelSetVTimerTime", 0x542AD630, since = 150) { sceKernelSetVTimerTime(it) }
		registerFunctionRaw("sceKernelCreateVpl", 0x56C039B5, since = 150) { sceKernelCreateVpl(it) }
		registerFunctionRaw("sceKernelGetThreadmanIdType", 0x57CF62DD, since = 150) { sceKernelGetThreadmanIdType(it) }
		registerFunctionRaw("sceKernelPollSema", 0x58B1F937, since = 150) { sceKernelPollSema(it) }
		registerFunctionRaw("sceKernelLockMutexCB", 0x5BF4DD27, since = 150) { sceKernelLockMutexCB(it) }
		registerFunctionRaw("sceKernelReferVTimerStatus", 0x5F32BEAA, since = 150) { sceKernelReferVTimerStatus(it) }
		registerFunctionRaw("sceKernelDeleteLwMutex", 0x60107536, since = 150) { sceKernelDeleteLwMutex(it) }
		registerFunctionRaw("sceKernelTerminateThread", 0x616403BA, since = 150) { sceKernelTerminateThread(it) }
		registerFunctionRaw("sceKernelTryAllocateFpl", 0x623AE665, since = 150) { sceKernelTryAllocateFpl(it) }
		registerFunctionRaw("sceKernelReferSystemStatus", 0x627E6F3A, since = 150) { sceKernelReferSystemStatus(it) }
		registerFunctionRaw("sceKernelReferThreadProfiler", 0x64D4540E, since = 150) { sceKernelReferThreadProfiler(it) }
		registerFunctionRaw("sceKernelSetAlarm", 0x6652B8CA, since = 150) { sceKernelSetAlarm(it) }
		registerFunctionRaw("sceKernelUnlockMutex", 0x6B30100F, since = 150) { sceKernelUnlockMutex(it) }
		registerFunctionRaw("sceKernelWaitSemaCB", 0x6D212BAC, since = 150) { sceKernelWaitSemaCB(it) }
		registerFunctionRaw("_sceKernelReturnFromCallback", 0x6E9EA350, since = 150) { _sceKernelReturnFromCallback(it) }
		registerFunctionRaw("ThreadManForUser_71040D5C", 0x71040D5C, since = 150) { ThreadManForUser_71040D5C(it) }
		registerFunctionRaw("sceKernelChangeThreadPriority", 0x71BC9871, since = 150) { sceKernelChangeThreadPriority(it) }
		registerFunctionRaw("sceKernelReleaseThreadEventHandler", 0x72F3C145, since = 150) { sceKernelReleaseThreadEventHandler(it) }
		registerFunctionRaw("sceKernelReferCallbackStatus", 0x730ED8BC, since = 150) { sceKernelReferCallbackStatus(it) }
		registerFunctionRaw("sceKernelReceiveMsgPipe", 0x74829B76, since = 150) { sceKernelReceiveMsgPipe(it) }
		registerFunctionRaw("sceKernelResumeThread", 0x75156E8F, since = 150) { sceKernelResumeThread(it) }
		registerFunctionRaw("sceKernelCreateMsgPipe", 0x7C0DC2A0, since = 150) { sceKernelCreateMsgPipe(it) }
		registerFunctionRaw("sceKernelSendMsgPipeCB", 0x7C41F2C2, since = 150) { sceKernelSendMsgPipeCB(it) }
		registerFunctionRaw("ThreadManForUser_7CFF8CF3", 0x7CFF8CF3, since = 150) { ThreadManForUser_7CFF8CF3(it) }
		registerFunctionRaw("sceKernelCancelAlarm", 0x7E65B999, since = 150) { sceKernelCancelAlarm(it) }
		registerFunctionRaw("sceKernelExitDeleteThread", 0x809CE29B, since = 150) { sceKernelExitDeleteThread(it) }
		registerFunctionRaw("sceKernelClearEventFlag", 0x812346E4, since = 150) { sceKernelClearEventFlag(it) }
		registerFunctionRaw("sceKernelCreateMbx", 0x8125221D, since = 150) { sceKernelCreateMbx(it) }
		registerFunctionRaw("sceKernelReferGlobalProfiler", 0x8218B4DD, since = 150) { sceKernelReferGlobalProfiler(it) }
		registerFunctionRaw("sceKernelWaitThreadEndCB", 0x840E8133, since = 150) { sceKernelWaitThreadEndCB(it) }
		registerFunctionRaw("sceKernelDeleteMbx", 0x86255ADA, since = 150) { sceKernelDeleteMbx(it) }
		registerFunctionRaw("ThreadManForUser_8672E3D0", 0x8672E3D0, since = 150) { ThreadManForUser_8672E3D0(it) }
		registerFunctionRaw("sceKernelSendMsgPipe", 0x876DBFAD, since = 150) { sceKernelSendMsgPipe(it) }
		registerFunctionRaw("sceKernelCancelReceiveMbx", 0x87D4DD36, since = 150) { sceKernelCancelReceiveMbx(it) }
		registerFunctionRaw("sceKernelCancelMutex", 0x87D9223C, since = 150) { sceKernelCancelMutex(it) }
		registerFunctionRaw("sceKernelTrySendMsgPipe", 0x884C9F90, since = 150) { sceKernelTrySendMsgPipe(it) }
		registerFunctionRaw("sceKernelDeleteVpl", 0x89B3D48C, since = 150) { sceKernelDeleteVpl(it) }
		registerFunctionRaw("sceKernelCancelSema", 0x8FFDF9A2, since = 150) { sceKernelCancelSema(it) }
		registerFunctionRaw("sceKernelRotateThreadReadyQueue", 0x912354A7, since = 150) { sceKernelRotateThreadReadyQueue(it) }
		registerFunctionRaw("sceKernelGetThreadmanIdList", 0x94416130, since = 150) { sceKernelGetThreadmanIdList(it) }
		registerFunctionRaw("sceKernelSuspendThread", 0x9944F31F, since = 150) { sceKernelSuspendThread(it) }
		registerFunctionRaw("sceKernelSleepThread", 0x9ACE131E, since = 150) { sceKernelSleepThread(it) }
		registerFunctionRaw("sceKernelDeleteThread", 0x9FA03CD3, since = 150) { sceKernelDeleteThread(it) }
		registerFunctionRaw("sceKernelReferEventFlagStatus", 0xA66B0120, since = 150) { sceKernelReferEventFlagStatus(it) }
		registerFunctionRaw("sceKernelCancelFpl", 0xA8AA591F, since = 150) { sceKernelCancelFpl(it) }
		registerFunctionRaw("sceKernelReferMbxStatus", 0xA8E8C846, since = 150) { sceKernelReferMbxStatus(it) }
		registerFunctionRaw("sceKernelReferMutexStatus", 0xA9C2CB9A, since = 150) { sceKernelReferMutexStatus(it) }
		registerFunctionRaw("sceKernelExitThread", 0xAA73C935, since = 150) { sceKernelExitThread(it) }
		registerFunctionRaw("sceKernelTryAllocateVpl", 0xAF36D708, since = 150) { sceKernelTryAllocateVpl(it) }
		registerFunctionRaw("sceKernelLockMutex", 0xB011B11F, since = 150) { sceKernelLockMutex(it) }
		registerFunctionRaw("sceKernelSetSysClockAlarm", 0xB2C25152, since = 150) { sceKernelSetSysClockAlarm(it) }
		registerFunctionRaw("sceKernelGetVTimerBase", 0xB3A59970, since = 150) { sceKernelGetVTimerBase(it) }
		registerFunctionRaw("sceKernelFreeVpl", 0xB736E9FF, since = 150) { sceKernelFreeVpl(it) }
		registerFunctionRaw("sceKernelGetVTimerBaseWide", 0xB7C18B77, since = 150) { sceKernelGetVTimerBaseWide(it) }
		registerFunctionRaw("sceKernelCreateMutex", 0xB7D098C6, since = 150) { sceKernelCreateMutex(it) }
		registerFunctionRaw("sceKernelCancelCallback", 0xBA4051D6, since = 150) { sceKernelCancelCallback(it) }
		registerFunctionRaw("sceKernelSysClock2USec", 0xBA6B92E2, since = 150) { sceKernelSysClock2USec(it) }
		registerFunctionRaw("sceKernelReferSemaStatus", 0xBC6FEBC5, since = 150) { sceKernelReferSemaStatus(it) }
		registerFunctionRaw("sceKernelDelaySysClockThread", 0xBD123D9E, since = 150) { sceKernelDelaySysClockThread(it) }
		registerFunctionRaw("sceKernelAllocateVpl", 0xBED27435, since = 150) { sceKernelAllocateVpl(it) }
		registerFunctionRaw("ThreadManForUser_BEED3A47", 0xBEED3A47, since = 150) { ThreadManForUser_BEED3A47(it) }
		registerFunctionRaw("sceKernelCreateFpl", 0xC07BB470, since = 150) { sceKernelCreateFpl(it) }
		registerFunctionRaw("sceKernelGetVTimerTimeWide", 0xC0B3FFD2, since = 150) { sceKernelGetVTimerTimeWide(it) }
		registerFunctionRaw("sceKernelNotifyCallback", 0xC11BA8C4, since = 150) { sceKernelNotifyCallback(it) }
		registerFunctionRaw("sceKernelStartVTimer", 0xC68D9437, since = 150) { sceKernelStartVTimer(it) }
		registerFunctionRaw("sceKernelUSec2SysClockWide", 0xC8CD158C, since = 150) { sceKernelUSec2SysClockWide(it) }
		registerFunctionRaw("sceKernelCancelEventFlag", 0xCD203292, since = 150) { sceKernelCancelEventFlag(it) }
		registerFunctionRaw("sceKernelStopVTimer", 0xD0AEEE87, since = 150) { sceKernelStopVTimer(it) }
		registerFunctionRaw("sceKernelCheckThreadStack", 0xD13BDE95, since = 150) { sceKernelCheckThreadStack(it) }
		registerFunctionRaw("sceKernelCancelVTimerHandler", 0xD2D615EF, since = 150) { sceKernelCancelVTimerHandler(it) }
		registerFunctionRaw("sceKernelWakeupThread", 0xD59EAD2F, since = 150) { sceKernelWakeupThread(it) }
		registerFunctionRaw("sceKernelCreateSema", 0xD6DA4BA1, since = 150) { sceKernelCreateSema(it) }
		registerFunctionRaw("sceKernelReferFplStatus", 0xD8199E4C, since = 150) { sceKernelReferFplStatus(it) }
		registerFunctionRaw("sceKernelSetVTimerHandler", 0xD8B299AE, since = 150) { sceKernelSetVTimerHandler(it) }
		registerFunctionRaw("sceKernelAllocateFpl", 0xD979E9BF, since = 150) { sceKernelAllocateFpl(it) }
		registerFunctionRaw("sceKernelReferAlarmStatus", 0xDAA3F564, since = 150) { sceKernelReferAlarmStatus(it) }
		registerFunctionRaw("sceKernelGetSystemTime", 0xDB738F35, since = 150) { sceKernelGetSystemTime(it) }
		registerFunctionRaw("sceKernelTryReceiveMsgPipe", 0xDF52098F, since = 150) { sceKernelTryReceiveMsgPipe(it) }
		registerFunctionRaw("sceKernelSysClock2USecWide", 0xE1619D7C, since = 150) { sceKernelSysClock2USecWide(it) }
		registerFunctionRaw("sceKernelAllocateFplCB", 0xE7282CB6, since = 150) { sceKernelAllocateFplCB(it) }
		registerFunctionRaw("sceKernelSendMbx", 0xE9B3061E, since = 150) { sceKernelSendMbx(it) }
		registerFunctionRaw("sceKernelChangeCurrentThreadAttr", 0xEA748E31, since = 150) { sceKernelChangeCurrentThreadAttr(it) }
		registerFunctionRaw("sceKernelAllocateVplCB", 0xEC0A693F, since = 150) { sceKernelAllocateVplCB(it) }
		registerFunctionRaw("sceKernelDeleteFpl", 0xED1410E0, since = 150) { sceKernelDeleteFpl(it) }
		registerFunctionRaw("sceKernelDeleteCallback", 0xEDBA5844, since = 150) { sceKernelDeleteCallback(it) }
		registerFunctionRaw("sceKernelDeleteEventFlag", 0xEF9E4C70, since = 150) { sceKernelDeleteEventFlag(it) }
		registerFunctionRaw("sceKernelDeleteMsgPipe", 0xF0B7DA1C, since = 150) { sceKernelDeleteMsgPipe(it) }
		registerFunctionRaw("sceKernelReceiveMbxCB", 0xF3986382, since = 150) { sceKernelReceiveMbxCB(it) }
		registerFunctionRaw("sceKernelFreeFpl", 0xF6414A71, since = 150) { sceKernelFreeFpl(it) }
		registerFunctionRaw("sceKernelDeleteMutex", 0xF8170FBE, since = 150) { sceKernelDeleteMutex(it) }
		registerFunctionRaw("sceKernelSetVTimerTimeWide", 0xFB6425C3, since = 150) { sceKernelSetVTimerTimeWide(it) }
		registerFunctionRaw("sceKernelReceiveMsgPipeCB", 0xFBFA697D, since = 150) { sceKernelReceiveMsgPipeCB(it) }
		registerFunctionRaw("sceKernelCancelWakeupThread", 0xFCCFAD26, since = 150) { sceKernelCancelWakeupThread(it) }
		registerFunctionRaw("sceKernelReferThreadRunStatus", 0xFFC36A14, since = 150) { sceKernelReferThreadRunStatus(it) }
	}
}

data class PspEventFlag(override val id: Int) : ResourceItem {
	var name: String = ""
	var attributes: Int = 0
	var bitPattern: Int = 0
	var optionsPtr: Ptr? = null
}
