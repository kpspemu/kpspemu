package com.soywiz.kpspemu.battery

import com.soywiz.korio.util.IdEnum
import com.soywiz.kpspemu.Emulator

class PspBattery(val emulator: Emulator) {
	var charging = false
	var percentage = 1.0

	val isLowBattery: Boolean = percentage < 0.22

	val chargingType: ChargingEnum = ChargingEnum(if (this.charging) 1 else 0)

	val iconStatus: BatteryStatusEnum
		get() = when {
			percentage < 0.15 -> BatteryStatusEnum.VeryLow
			percentage < 0.30 -> BatteryStatusEnum.Low
			percentage < 0.80 -> BatteryStatusEnum.PartiallyFilled
			else -> BatteryStatusEnum.FullyFilled
		}
}

enum class ChargingEnum(override val id: Int) : IdEnum {
	NotCharging(0),
	Charging(1);

	companion object : IdEnum.SmallCompanion<ChargingEnum>(values())
}

enum class BatteryStatusEnum(override val id: Int) : IdEnum {
	VeryLow(0),
	Low(1),
	PartiallyFilled(2),
	FullyFilled(3);

	companion object : IdEnum.SmallCompanion<BatteryStatusEnum>(values())
}