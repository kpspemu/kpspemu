package com.soywiz.kpspemu.battery

import com.soywiz.korio.util.*
import com.soywiz.kpspemu.*

class PspBattery(val emulator: Emulator) {
    var charging = false
    var chargedRatio = 1.0
    var lifetimeSeconds: Double = 8.0 * 3600.0
    var batteryExists: Boolean = true

    val volt: Int = 4135
    val temp: Int = 28
    val level: Double get() = chargedRatio
    val percentage: Double get() = chargedRatio
    val isLowBattery: Boolean = chargedRatio < 0.22

    val chargingType: ChargingEnum = ChargingEnum(if (this.charging) 1 else 0)

    val iconStatus: BatteryStatusEnum
        get() = when {
            chargedRatio < 0.15 -> BatteryStatusEnum.VeryLow
            chargedRatio < 0.30 -> BatteryStatusEnum.Low
            chargedRatio < 0.80 -> BatteryStatusEnum.PartiallyFilled
            else -> BatteryStatusEnum.FullyFilled
        }

    fun reset() {
        charging = false
        chargedRatio = 1.0
        lifetimeSeconds = 8.0 * 3600.0
        batteryExists = true
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
