package com.soywiz.kpspemu.hle.modules

import com.soywiz.klock.DateTime
import com.soywiz.klock.Month
import com.soywiz.korio.util.clamp
import com.soywiz.kpspemu.Emulator
import com.soywiz.kpspemu.cpu.CpuState
import com.soywiz.kpspemu.hle.SceModule
import com.soywiz.kpspemu.hle.manager.ScePspDateTime
import com.soywiz.kpspemu.mem.*
import com.soywiz.kpspemu.timeManager

@Suppress("UNUSED_PARAMETER")
class sceRtc(emulator: Emulator) : SceModule(emulator, "sceRtc", 0x40010011, "rtc.prx", "sceRTC_Service") {
	fun sceRtcGetCurrentTick(ptr: Ptr): Int = 0.apply { ptr.sdw(0, timeManager.getTimeInMicroseconds()) }
	fun sceRtcGetTickResolution(): Int = 1000000
	fun sceRtcGetDayOfWeek(year: Int, month: Int, day: Int): Int = DateTime(year, month, day).dayOfWeekInt
	fun sceRtcGetDaysInMonth(year: Int, month: Int): Int = Month.days(month, year)
	fun sceRtcSetTick(datePtr: Ptr, ticksPtr: Ptr): Int {
		val ticks = ticksPtr.ldw(0)
		val time = ScePspDateTime(DateTime(ticks))
		datePtr.write(ScePspDateTime, time)
		return 0
	}

	fun sceRtcGetTick(datePtr: Ptr, ticksPtr: Ptr64): Int {
		val date = ScePspDateTime.read(datePtr.openSync())
		ticksPtr.set(date.date.unix * 1000)
		return 0
	}

	fun sceRtcGetCurrentClock(date: Ptr, timezone: Int): Int {
		if (date.isNotNull) ScePspDateTime(DateTime.now().addOffset(timezone.clamp(-600000, +600000))).write(date.openSync())
		return 0
	}

	fun sceRtcGetCurrentClockLocalTime(date: Ptr): Int {
		if (date.isNotNull) ScePspDateTime(DateTime.now().toLocal()).write(date.openSync())
		return 0
	}

	fun sceRtcTickAddMicroseconds(dst: Ptr64, src: Ptr64, count: Long): Int {
		dst.set(src.get() + count)
		return 0
	}

	fun sceRtcCompareTick(tick1: Ptr64, tick2: Ptr64): Int {
		return tick1.get().compareTo(tick2.get())
	}

	fun sceRtcTickAddTicks(dest: Ptr64, src: Ptr64, num: Long): Int {
		dest.set(src.get() + num)
		return 0
	}

	fun sceRtcGetAccumulativeTime(cpu: CpuState): Unit = UNIMPLEMENTED(0x011F03C1)
	fun sceRtcGetAccumlativeTime(cpu: CpuState): Unit = UNIMPLEMENTED(0x029CA3B3)
	fun sceRtcFormatRFC3339(cpu: CpuState): Unit = UNIMPLEMENTED(0x0498FB3C)
	fun sceRtcSetTime64_t(cpu: CpuState): Unit = UNIMPLEMENTED(0x1909C99B)
	fun sceRtcGetLastReincarnatedTime(cpu: CpuState): Unit = UNIMPLEMENTED(0x203CEB0D)
	fun sceRtcTickAddHours(cpu: CpuState): Unit = UNIMPLEMENTED(0x26D7A24A)
	fun sceRtcGetTime_t(cpu: CpuState): Unit = UNIMPLEMENTED(0x27C4594C)
	fun sceRtcFormatRFC3339LocalTime(cpu: CpuState): Unit = UNIMPLEMENTED(0x27F98543)
	fun sceRtcParseRFC3339(cpu: CpuState): Unit = UNIMPLEMENTED(0x28E1E988)
	fun sceRtcConvertUtcToLocalTime(cpu: CpuState): Unit = UNIMPLEMENTED(0x34885E0D)
	fun sceRtcGetDosTime(cpu: CpuState): Unit = UNIMPLEMENTED(0x36075567)
	fun sceRtcSetTime_t(cpu: CpuState): Unit = UNIMPLEMENTED(0x3A807CC8)
	fun sceRtcIsLeapYear(cpu: CpuState): Unit = UNIMPLEMENTED(0x42307A17)
	fun sceRtcTickAddYears(cpu: CpuState): Unit = UNIMPLEMENTED(0x42842C77)
	fun sceRtcCheckValid(cpu: CpuState): Unit = UNIMPLEMENTED(0x4B1B5E82)
	fun sceRtcGetLastAdjustedTime(cpu: CpuState): Unit = UNIMPLEMENTED(0x62685E98)
	fun sceRtcUnregisterCallback(cpu: CpuState): Unit = UNIMPLEMENTED(0x6A676D2D)
	fun sceRtcConvertLocalTimeToUTC(cpu: CpuState): Unit = UNIMPLEMENTED(0x779242A2)
	fun sceRtcSetWin32FileTime(cpu: CpuState): Unit = UNIMPLEMENTED(0x7ACE4C04)
	fun sceRtc_7D1FBED3(cpu: CpuState): Unit = UNIMPLEMENTED(0x7D1FBED3)
	fun sceRtcFormatRFC2822LocalTime(cpu: CpuState): Unit = UNIMPLEMENTED(0x7DE6711B)
	fun sceRtcIsAlarmed(cpu: CpuState): Unit = UNIMPLEMENTED(0x81FCDA34)
	fun sceRtc_A93CF7D8(cpu: CpuState): Unit = UNIMPLEMENTED(0xA93CF7D8)
	fun sceRtc_C2DDBEB5(cpu: CpuState): Unit = UNIMPLEMENTED(0xC2DDBEB5)
	fun sceRtcFormatRFC2822(cpu: CpuState): Unit = UNIMPLEMENTED(0xC663B3B9)
	fun sceRtcTickAddWeeks(cpu: CpuState): Unit = UNIMPLEMENTED(0xCF3A2CA8)
	fun sceRtcGetWin32FileTime(cpu: CpuState): Unit = UNIMPLEMENTED(0xCF561893)
	fun sceRtcTickAddMonths(cpu: CpuState): Unit = UNIMPLEMENTED(0xDBF74F1B)
	fun sceRtcParseDateTime(cpu: CpuState): Unit = UNIMPLEMENTED(0xDFBC5F16)
	fun sceRtcGetTime64_t(cpu: CpuState): Unit = UNIMPLEMENTED(0xE1C93E47)
	fun sceRtcTickAddDays(cpu: CpuState): Unit = UNIMPLEMENTED(0xE51B4B7A)
	fun sceRtcTickAddMinutes(cpu: CpuState): Unit = UNIMPLEMENTED(0xE6605BCA)
	fun sceRtcSetDosTime(cpu: CpuState): Unit = UNIMPLEMENTED(0xF006F264)
	fun sceRtcTickAddSeconds(cpu: CpuState): Unit = UNIMPLEMENTED(0xF2A4AFE5)
	fun sceRtc_F5FCC995(cpu: CpuState): Unit = UNIMPLEMENTED(0xF5FCC995)
	fun sceRtcRegisterCallback(cpu: CpuState): Unit = UNIMPLEMENTED(0xFB3B18CD)

	override fun registerModule() {
		registerFunctionInt("sceRtcGetCurrentTick", 0x3F7AD767, since = 150) { sceRtcGetCurrentTick(ptr) }
		registerFunctionInt("sceRtcGetTickResolution", 0xC41C2853, since = 150) { sceRtcGetTickResolution() }
		registerFunctionInt("sceRtcGetDayOfWeek", 0x57726BC1, since = 150) { sceRtcGetDayOfWeek(int, int, int) }
		registerFunctionInt("sceRtcGetDaysInMonth", 0x05EF322C, since = 150) { sceRtcGetDaysInMonth(int, int) }
		registerFunctionInt("sceRtcSetTick", 0x7ED29E40, since = 150) { sceRtcSetTick(ptr, ptr) }
		registerFunctionInt("sceRtcGetTick", 0x6FF40ACC, since = 150) { sceRtcGetTick(ptr, ptr64) }
		registerFunctionInt("sceRtcGetCurrentClock", 0x4CFA57B0, since = 150) { sceRtcGetCurrentClock(ptr, int) }
		registerFunctionInt("sceRtcGetCurrentClockLocalTime", 0xE7C27D1B, since = 150) { sceRtcGetCurrentClockLocalTime(ptr) }
		registerFunctionInt("sceRtcTickAddMicroseconds", 0x26D25A5D, since = 150) { sceRtcTickAddMicroseconds(ptr64, ptr64, long) }
		registerFunctionInt("sceRtcCompareTick", 0x9ED0AE87, since = 150) { sceRtcCompareTick(ptr64, ptr64) }
		registerFunctionInt("sceRtcTickAddTicks", 0x44F45E05, since = 150) { sceRtcTickAddTicks(ptr64, ptr64, long) }

		registerFunctionRaw("sceRtcGetAccumulativeTime", 0x011F03C1, since = 150) { sceRtcGetAccumulativeTime(it) }
		registerFunctionRaw("sceRtcGetAccumlativeTime", 0x029CA3B3, since = 150) { sceRtcGetAccumlativeTime(it) }
		registerFunctionRaw("sceRtcFormatRFC3339", 0x0498FB3C, since = 150) { sceRtcFormatRFC3339(it) }
		registerFunctionRaw("sceRtcSetTime64_t", 0x1909C99B, since = 150) { sceRtcSetTime64_t(it) }
		registerFunctionRaw("sceRtcGetLastReincarnatedTime", 0x203CEB0D, since = 150) { sceRtcGetLastReincarnatedTime(it) }
		registerFunctionRaw("sceRtcTickAddHours", 0x26D7A24A, since = 150) { sceRtcTickAddHours(it) }
		registerFunctionRaw("sceRtcGetTime_t", 0x27C4594C, since = 150) { sceRtcGetTime_t(it) }
		registerFunctionRaw("sceRtcFormatRFC3339LocalTime", 0x27F98543, since = 150) { sceRtcFormatRFC3339LocalTime(it) }
		registerFunctionRaw("sceRtcParseRFC3339", 0x28E1E988, since = 150) { sceRtcParseRFC3339(it) }
		registerFunctionRaw("sceRtcConvertUtcToLocalTime", 0x34885E0D, since = 150) { sceRtcConvertUtcToLocalTime(it) }
		registerFunctionRaw("sceRtcGetDosTime", 0x36075567, since = 150) { sceRtcGetDosTime(it) }
		registerFunctionRaw("sceRtcSetTime_t", 0x3A807CC8, since = 150) { sceRtcSetTime_t(it) }
		registerFunctionRaw("sceRtcIsLeapYear", 0x42307A17, since = 150) { sceRtcIsLeapYear(it) }
		registerFunctionRaw("sceRtcTickAddYears", 0x42842C77, since = 150) { sceRtcTickAddYears(it) }
		registerFunctionRaw("sceRtcCheckValid", 0x4B1B5E82, since = 150) { sceRtcCheckValid(it) }
		registerFunctionRaw("sceRtcGetLastAdjustedTime", 0x62685E98, since = 150) { sceRtcGetLastAdjustedTime(it) }
		registerFunctionRaw("sceRtcUnregisterCallback", 0x6A676D2D, since = 150) { sceRtcUnregisterCallback(it) }
		registerFunctionRaw("sceRtcConvertLocalTimeToUTC", 0x779242A2, since = 150) { sceRtcConvertLocalTimeToUTC(it) }
		registerFunctionRaw("sceRtcSetWin32FileTime", 0x7ACE4C04, since = 150) { sceRtcSetWin32FileTime(it) }
		registerFunctionRaw("sceRtc_7D1FBED3", 0x7D1FBED3, since = 150) { sceRtc_7D1FBED3(it) }
		registerFunctionRaw("sceRtcFormatRFC2822LocalTime", 0x7DE6711B, since = 150) { sceRtcFormatRFC2822LocalTime(it) }
		registerFunctionRaw("sceRtcIsAlarmed", 0x81FCDA34, since = 150) { sceRtcIsAlarmed(it) }
		registerFunctionRaw("sceRtc_A93CF7D8", 0xA93CF7D8, since = 150) { sceRtc_A93CF7D8(it) }
		registerFunctionRaw("sceRtc_C2DDBEB5", 0xC2DDBEB5, since = 150) { sceRtc_C2DDBEB5(it) }
		registerFunctionRaw("sceRtcFormatRFC2822", 0xC663B3B9, since = 150) { sceRtcFormatRFC2822(it) }
		registerFunctionRaw("sceRtcTickAddWeeks", 0xCF3A2CA8, since = 150) { sceRtcTickAddWeeks(it) }
		registerFunctionRaw("sceRtcGetWin32FileTime", 0xCF561893, since = 150) { sceRtcGetWin32FileTime(it) }
		registerFunctionRaw("sceRtcTickAddMonths", 0xDBF74F1B, since = 150) { sceRtcTickAddMonths(it) }
		registerFunctionRaw("sceRtcParseDateTime", 0xDFBC5F16, since = 150) { sceRtcParseDateTime(it) }
		registerFunctionRaw("sceRtcGetTime64_t", 0xE1C93E47, since = 150) { sceRtcGetTime64_t(it) }
		registerFunctionRaw("sceRtcTickAddDays", 0xE51B4B7A, since = 150) { sceRtcTickAddDays(it) }
		registerFunctionRaw("sceRtcTickAddMinutes", 0xE6605BCA, since = 150) { sceRtcTickAddMinutes(it) }
		registerFunctionRaw("sceRtcSetDosTime", 0xF006F264, since = 150) { sceRtcSetDosTime(it) }
		registerFunctionRaw("sceRtcTickAddSeconds", 0xF2A4AFE5, since = 150) { sceRtcTickAddSeconds(it) }
		registerFunctionRaw("sceRtc_F5FCC995", 0xF5FCC995, since = 150) { sceRtc_F5FCC995(it) }
		registerFunctionRaw("sceRtcRegisterCallback", 0xFB3B18CD, since = 150) { sceRtcRegisterCallback(it) }
	}
}
