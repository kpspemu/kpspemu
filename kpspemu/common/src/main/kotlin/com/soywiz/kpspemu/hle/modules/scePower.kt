package com.soywiz.kpspemu.hle.modules

import com.soywiz.kpspemu.Emulator
import com.soywiz.kpspemu.cpu.CpuState
import com.soywiz.kpspemu.hle.SceModule
import kotlin.math.floor

@Suppress("UNUSED_PARAMETER")
class scePower(emulator: Emulator) : SceModule(emulator, "scePower", 0x40010011, "power.prx", "scePower_Service") {
	// 222/111
	// 333/166

	private var cpuMult = 511f // 222mhz
	private var pllFreq = 222f
	private var busFreq = 111f // MAX BUS: 166

	fun _getCpuMult(): Float = (0.43444227005871f * (this.busFreq / 111f))
	fun _getCpuFreq(): Float = this.cpuMult * this._getCpuMult()

	fun _setCpuFreq(cpuFreq: Int): Unit {
		this.cpuMult = when {
			cpuFreq > 222 -> this.cpuMult // TODO: necessary until integer arithmetic to avoid it failing
			cpuFreq == 222 -> 511f;
			else -> cpuFreq / this._getCpuMult();
		}
	}

	fun scePowerGetCpuClockFrequencyInt(): Int = _getCpuFreq().toInt()

	fun scePowerGetResumeCount(cpu: CpuState): Unit = UNIMPLEMENTED(0x0074EF9B)
	fun scePowerRequestColdReset(cpu: CpuState): Unit = UNIMPLEMENTED(0x0442D852)
	fun scePowerRegisterCallback(cpu: CpuState): Unit = UNIMPLEMENTED(0x04B7766E)
	fun scePowerIsBatteryExist(cpu: CpuState): Unit = UNIMPLEMENTED(0x0AFD0D8B)
	fun scePowerSetPowerSwMode(cpu: CpuState): Unit = UNIMPLEMENTED(0x0CD21B1F)
	fun scePowerGetPowerSwMode(cpu: CpuState): Unit = UNIMPLEMENTED(0x165CE085)
	fun scePowerIsBatteryCharging(cpu: CpuState): Unit = UNIMPLEMENTED(0x1E490401)
	fun scePowerGetBatteryLifePercent(cpu: CpuState): Unit = UNIMPLEMENTED(0x2085D15D)
	fun scePowerGetInnerTemp(cpu: CpuState): Unit = UNIMPLEMENTED(0x23436A4A)
	fun scePowerVolatileMemLock(cpu: CpuState): Unit = UNIMPLEMENTED(0x23C31FFE)
	fun scePowerBatteryUpdateInfo(cpu: CpuState): Unit = UNIMPLEMENTED(0x27F3292C)
	fun scePower_2875994B(cpu: CpuState): Unit = UNIMPLEMENTED(0x2875994B)
	fun scePowerGetBatteryTemp(cpu: CpuState): Unit = UNIMPLEMENTED(0x28E12023)
	fun scePower_2B51FE2F(cpu: CpuState): Unit = UNIMPLEMENTED(0x2B51FE2F)
	fun scePowerRequestStandby(cpu: CpuState): Unit = UNIMPLEMENTED(0x2B7C7CF4)
	fun scePowerGetPllClockFrequencyInt(cpu: CpuState): Unit = UNIMPLEMENTED(0x34F9C463)
	fun scePowerWaitRequestCompletion(cpu: CpuState): Unit = UNIMPLEMENTED(0x3951AF53)
	fun scePowerGetBacklightMaximum(cpu: CpuState): Unit = UNIMPLEMENTED(0x442BFBAC)
	fun scePowerGetBusClockFrequency(cpu: CpuState): Unit = UNIMPLEMENTED(0x478FE6F5)
	fun scePowerGetBatteryVolt(cpu: CpuState): Unit = UNIMPLEMENTED(0x483CE86B)
	fun scePower_545A7F3C(cpu: CpuState): Unit = UNIMPLEMENTED(0x545A7F3C)
	fun scePowerSetClockFrequency(cpu: CpuState): Unit = UNIMPLEMENTED(0x737486F2)
	fun scePowerIsSuspendRequired(cpu: CpuState): Unit = UNIMPLEMENTED(0x78A1A796)
	fun scePowerIdleTimerEnable(cpu: CpuState): Unit = UNIMPLEMENTED(0x7F30B3B1)
	fun scePowerIsRequest(cpu: CpuState): Unit = UNIMPLEMENTED(0x7FA406DD)
	fun scePowerSetCpuClockFrequency(cpu: CpuState): Unit = UNIMPLEMENTED(0x843FBF43)
	fun scePowerGetBatteryElec(cpu: CpuState): Unit = UNIMPLEMENTED(0x862AE1A6)
	fun scePowerIsPowerOnline(cpu: CpuState): Unit = UNIMPLEMENTED(0x87440F5E)
	fun scePowerGetBatteryLifeTime(cpu: CpuState): Unit = UNIMPLEMENTED(0x8EFB3FA2)
	fun scePowerGetBatteryRemainCapacity(cpu: CpuState): Unit = UNIMPLEMENTED(0x94F5A53F)
	fun scePowerIdleTimerDisable(cpu: CpuState): Unit = UNIMPLEMENTED(0x972CE941)
	fun scePowerGetBusClockFrequencyFloat(cpu: CpuState): Unit = UNIMPLEMENTED(0x9BADB3EB)
	fun scePower_A4E93389(cpu: CpuState): Unit = UNIMPLEMENTED(0xA4E93389)
	fun scePowerSetCallbackMode(cpu: CpuState): Unit = UNIMPLEMENTED(0xA9D22232)
	fun scePowerRequestSuspend(cpu: CpuState): Unit = UNIMPLEMENTED(0xAC32C9CC)
	fun scePowerGetCpuClockFrequencyFloat(cpu: CpuState): Unit = UNIMPLEMENTED(0xB1A52C83)
	fun scePowerVolatileMemUnlock(cpu: CpuState): Unit = UNIMPLEMENTED(0xB3EDD801)
	fun scePowerGetBatteryChargingStatus(cpu: CpuState): Unit = UNIMPLEMENTED(0xB4432BC8)
	fun scePowerSetBusClockFrequency(cpu: CpuState): Unit = UNIMPLEMENTED(0xB8D7B3FB)
	fun scePowerGetLowBatteryCapacity(cpu: CpuState): Unit = UNIMPLEMENTED(0xB999184C)
	fun scePowerGetCallbackMode(cpu: CpuState): Unit = UNIMPLEMENTED(0xBAFA3DF0)
	fun scePowerGetBusClockFrequencyInt(cpu: CpuState): Unit = UNIMPLEMENTED(0xBD681969)
	fun scePowerUnlock(cpu: CpuState): Unit = UNIMPLEMENTED(0xCA3D34C1)
	fun scePowerGetBatteryChargeCycle(cpu: CpuState): Unit = UNIMPLEMENTED(0xCB49F5CE)
	fun scePowerIsLowBattery(cpu: CpuState): Unit = UNIMPLEMENTED(0xD3075926)
	fun scePowerLock(cpu: CpuState): Unit = UNIMPLEMENTED(0xD6D016EF)
	fun scePowerCancelRequest(cpu: CpuState): Unit = UNIMPLEMENTED(0xDB62C9CF)
	fun scePowerUnregitserCallback(cpu: CpuState): Unit = UNIMPLEMENTED(0xDB9D28DD)
	fun scePowerUnregisterCallback(cpu: CpuState): Unit = UNIMPLEMENTED(0xDFA8BAF8)
	fun scePower_E8E4E204(cpu: CpuState): Unit = UNIMPLEMENTED(0xE8E4E204)
	fun scePowerGetPllClockFrequencyFloat(cpu: CpuState): Unit = UNIMPLEMENTED(0xEA382A27)
	fun scePower_EBD177D6(cpu: CpuState): Unit = UNIMPLEMENTED(0xEBD177D6)
	fun scePowerGetIdleTimer(cpu: CpuState): Unit = UNIMPLEMENTED(0xEDC13FE5)
	fun scePowerTick(cpu: CpuState): Unit = UNIMPLEMENTED(0xEFD3C963)
	fun scePowerVolatileMemTryLock(cpu: CpuState): Unit = UNIMPLEMENTED(0xFA97A599)
	fun scePowerGetBatteryFullCapacity(cpu: CpuState): Unit = UNIMPLEMENTED(0xFD18A0FF)
	fun scePowerGetCpuClockFrequency(cpu: CpuState): Unit = UNIMPLEMENTED(0xFEE03A2F)


	override fun registerModule() {
		registerFunctionInt("scePowerGetCpuClockFrequencyInt", 0xFDB5BFE9, since = 150) { scePowerGetCpuClockFrequencyInt() }

		registerFunctionRaw("scePowerGetResumeCount", 0x0074EF9B, since = 150) { scePowerGetResumeCount(it) }
		registerFunctionRaw("scePowerRequestColdReset", 0x0442D852, since = 150) { scePowerRequestColdReset(it) }
		registerFunctionRaw("scePowerRegisterCallback", 0x04B7766E, since = 150) { scePowerRegisterCallback(it) }
		registerFunctionRaw("scePowerIsBatteryExist", 0x0AFD0D8B, since = 150) { scePowerIsBatteryExist(it) }
		registerFunctionRaw("scePowerSetPowerSwMode", 0x0CD21B1F, since = 150) { scePowerSetPowerSwMode(it) }
		registerFunctionRaw("scePowerGetPowerSwMode", 0x165CE085, since = 150) { scePowerGetPowerSwMode(it) }
		registerFunctionRaw("scePowerIsBatteryCharging", 0x1E490401, since = 150) { scePowerIsBatteryCharging(it) }
		registerFunctionRaw("scePowerGetBatteryLifePercent", 0x2085D15D, since = 150) { scePowerGetBatteryLifePercent(it) }
		registerFunctionRaw("scePowerGetInnerTemp", 0x23436A4A, since = 150) { scePowerGetInnerTemp(it) }
		registerFunctionRaw("scePowerVolatileMemLock", 0x23C31FFE, since = 150) { scePowerVolatileMemLock(it) }
		registerFunctionRaw("scePowerBatteryUpdateInfo", 0x27F3292C, since = 150) { scePowerBatteryUpdateInfo(it) }
		registerFunctionRaw("scePower_2875994B", 0x2875994B, since = 150) { scePower_2875994B(it) }
		registerFunctionRaw("scePowerGetBatteryTemp", 0x28E12023, since = 150) { scePowerGetBatteryTemp(it) }
		registerFunctionRaw("scePower_2B51FE2F", 0x2B51FE2F, since = 150) { scePower_2B51FE2F(it) }
		registerFunctionRaw("scePowerRequestStandby", 0x2B7C7CF4, since = 150) { scePowerRequestStandby(it) }
		registerFunctionRaw("scePowerGetPllClockFrequencyInt", 0x34F9C463, since = 150) { scePowerGetPllClockFrequencyInt(it) }
		registerFunctionRaw("scePowerWaitRequestCompletion", 0x3951AF53, since = 150) { scePowerWaitRequestCompletion(it) }
		registerFunctionRaw("scePowerGetBacklightMaximum", 0x442BFBAC, since = 150) { scePowerGetBacklightMaximum(it) }
		registerFunctionRaw("scePowerGetBusClockFrequency", 0x478FE6F5, since = 150) { scePowerGetBusClockFrequency(it) }
		registerFunctionRaw("scePowerGetBatteryVolt", 0x483CE86B, since = 150) { scePowerGetBatteryVolt(it) }
		registerFunctionRaw("scePower_545A7F3C", 0x545A7F3C, since = 150) { scePower_545A7F3C(it) }
		registerFunctionRaw("scePowerSetClockFrequency", 0x737486F2, since = 150) { scePowerSetClockFrequency(it) }
		registerFunctionRaw("scePowerIsSuspendRequired", 0x78A1A796, since = 150) { scePowerIsSuspendRequired(it) }
		registerFunctionRaw("scePowerIdleTimerEnable", 0x7F30B3B1, since = 150) { scePowerIdleTimerEnable(it) }
		registerFunctionRaw("scePowerIsRequest", 0x7FA406DD, since = 150) { scePowerIsRequest(it) }
		registerFunctionRaw("scePowerSetCpuClockFrequency", 0x843FBF43, since = 150) { scePowerSetCpuClockFrequency(it) }
		registerFunctionRaw("scePowerGetBatteryElec", 0x862AE1A6, since = 150) { scePowerGetBatteryElec(it) }
		registerFunctionRaw("scePowerIsPowerOnline", 0x87440F5E, since = 150) { scePowerIsPowerOnline(it) }
		registerFunctionRaw("scePowerGetBatteryLifeTime", 0x8EFB3FA2, since = 150) { scePowerGetBatteryLifeTime(it) }
		registerFunctionRaw("scePowerGetBatteryRemainCapacity", 0x94F5A53F, since = 150) { scePowerGetBatteryRemainCapacity(it) }
		registerFunctionRaw("scePowerIdleTimerDisable", 0x972CE941, since = 150) { scePowerIdleTimerDisable(it) }
		registerFunctionRaw("scePowerGetBusClockFrequencyFloat", 0x9BADB3EB, since = 150) { scePowerGetBusClockFrequencyFloat(it) }
		registerFunctionRaw("scePower_A4E93389", 0xA4E93389, since = 150) { scePower_A4E93389(it) }
		registerFunctionRaw("scePowerSetCallbackMode", 0xA9D22232, since = 150) { scePowerSetCallbackMode(it) }
		registerFunctionRaw("scePowerRequestSuspend", 0xAC32C9CC, since = 150) { scePowerRequestSuspend(it) }
		registerFunctionRaw("scePowerGetCpuClockFrequencyFloat", 0xB1A52C83, since = 150) { scePowerGetCpuClockFrequencyFloat(it) }
		registerFunctionRaw("scePowerVolatileMemUnlock", 0xB3EDD801, since = 150) { scePowerVolatileMemUnlock(it) }
		registerFunctionRaw("scePowerGetBatteryChargingStatus", 0xB4432BC8, since = 150) { scePowerGetBatteryChargingStatus(it) }
		registerFunctionRaw("scePowerSetBusClockFrequency", 0xB8D7B3FB, since = 150) { scePowerSetBusClockFrequency(it) }
		registerFunctionRaw("scePowerGetLowBatteryCapacity", 0xB999184C, since = 150) { scePowerGetLowBatteryCapacity(it) }
		registerFunctionRaw("scePowerGetCallbackMode", 0xBAFA3DF0, since = 150) { scePowerGetCallbackMode(it) }
		registerFunctionRaw("scePowerGetBusClockFrequencyInt", 0xBD681969, since = 150) { scePowerGetBusClockFrequencyInt(it) }
		registerFunctionRaw("scePowerUnlock", 0xCA3D34C1, since = 150) { scePowerUnlock(it) }
		registerFunctionRaw("scePowerGetBatteryChargeCycle", 0xCB49F5CE, since = 150) { scePowerGetBatteryChargeCycle(it) }
		registerFunctionRaw("scePowerIsLowBattery", 0xD3075926, since = 150) { scePowerIsLowBattery(it) }
		registerFunctionRaw("scePowerLock", 0xD6D016EF, since = 150) { scePowerLock(it) }
		registerFunctionRaw("scePowerCancelRequest", 0xDB62C9CF, since = 150) { scePowerCancelRequest(it) }
		registerFunctionRaw("scePowerUnregitserCallback", 0xDB9D28DD, since = 150) { scePowerUnregitserCallback(it) }
		registerFunctionRaw("scePowerUnregisterCallback", 0xDFA8BAF8, since = 150) { scePowerUnregisterCallback(it) }
		registerFunctionRaw("scePower_E8E4E204", 0xE8E4E204, since = 150) { scePower_E8E4E204(it) }
		registerFunctionRaw("scePowerGetPllClockFrequencyFloat", 0xEA382A27, since = 150) { scePowerGetPllClockFrequencyFloat(it) }
		registerFunctionRaw("scePower_EBD177D6", 0xEBD177D6, since = 150) { scePower_EBD177D6(it) }
		registerFunctionRaw("scePowerGetIdleTimer", 0xEDC13FE5, since = 150) { scePowerGetIdleTimer(it) }
		registerFunctionRaw("scePowerTick", 0xEFD3C963, since = 150) { scePowerTick(it) }
		registerFunctionRaw("scePowerVolatileMemTryLock", 0xFA97A599, since = 150) { scePowerVolatileMemTryLock(it) }
		registerFunctionRaw("scePowerGetBatteryFullCapacity", 0xFD18A0FF, since = 150) { scePowerGetBatteryFullCapacity(it) }
		registerFunctionRaw("scePowerGetCpuClockFrequency", 0xFEE03A2F, since = 150) { scePowerGetCpuClockFrequency(it) }
	}
}
