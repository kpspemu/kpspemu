package com.soywiz.kpspemu.hle.modules


import com.soywiz.kpspemu.cpu.CpuState
import com.soywiz.kpspemu.hle.SceModule


class sceRtc : SceModule("sceRtc", 0x40010011, "rtc.prx", "sceRTC_Service") {
	fun sceRtcGetAccumulativeTime(cpu: CpuState): Unit = UNIMPLEMENTED(0x011F03C1)
	fun sceRtcGetAccumlativeTime(cpu: CpuState): Unit = UNIMPLEMENTED(0x029CA3B3)
	fun sceRtcFormatRFC3339(cpu: CpuState): Unit = UNIMPLEMENTED(0x0498FB3C)
	fun sceRtcGetDaysInMonth(cpu: CpuState): Unit = UNIMPLEMENTED(0x05EF322C)
	fun sceRtcSetTime64_t(cpu: CpuState): Unit = UNIMPLEMENTED(0x1909C99B)
	fun sceRtcGetLastReincarnatedTime(cpu: CpuState): Unit = UNIMPLEMENTED(0x203CEB0D)
	fun sceRtcTickAddMicroseconds(cpu: CpuState): Unit = UNIMPLEMENTED(0x26D25A5D)
	fun sceRtcTickAddHours(cpu: CpuState): Unit = UNIMPLEMENTED(0x26D7A24A)
	fun sceRtcGetTime_t(cpu: CpuState): Unit = UNIMPLEMENTED(0x27C4594C)
	fun sceRtcFormatRFC3339LocalTime(cpu: CpuState): Unit = UNIMPLEMENTED(0x27F98543)
	fun sceRtcParseRFC3339(cpu: CpuState): Unit = UNIMPLEMENTED(0x28E1E988)
	fun sceRtcConvertUtcToLocalTime(cpu: CpuState): Unit = UNIMPLEMENTED(0x34885E0D)
	fun sceRtcGetDosTime(cpu: CpuState): Unit = UNIMPLEMENTED(0x36075567)
	fun sceRtcSetTime_t(cpu: CpuState): Unit = UNIMPLEMENTED(0x3A807CC8)
	fun sceRtcGetCurrentTick(cpu: CpuState): Unit = UNIMPLEMENTED(0x3F7AD767)
	fun sceRtcIsLeapYear(cpu: CpuState): Unit = UNIMPLEMENTED(0x42307A17)
	fun sceRtcTickAddYears(cpu: CpuState): Unit = UNIMPLEMENTED(0x42842C77)
	fun sceRtcTickAddTicks(cpu: CpuState): Unit = UNIMPLEMENTED(0x44F45E05)
	fun sceRtcCheckValid(cpu: CpuState): Unit = UNIMPLEMENTED(0x4B1B5E82)
	fun sceRtcGetCurrentClock(cpu: CpuState): Unit = UNIMPLEMENTED(0x4CFA57B0)
	fun sceRtcGetDayOfWeek(cpu: CpuState): Unit = UNIMPLEMENTED(0x57726BC1)
	fun sceRtcGetLastAdjustedTime(cpu: CpuState): Unit = UNIMPLEMENTED(0x62685E98)
	fun sceRtcUnregisterCallback(cpu: CpuState): Unit = UNIMPLEMENTED(0x6A676D2D)
	fun sceRtcGetTick(cpu: CpuState): Unit = UNIMPLEMENTED(0x6FF40ACC)
	fun sceRtcConvertLocalTimeToUTC(cpu: CpuState): Unit = UNIMPLEMENTED(0x779242A2)
	fun sceRtcSetWin32FileTime(cpu: CpuState): Unit = UNIMPLEMENTED(0x7ACE4C04)
	fun sceRtc_7D1FBED3(cpu: CpuState): Unit = UNIMPLEMENTED(0x7D1FBED3)
	fun sceRtcFormatRFC2822LocalTime(cpu: CpuState): Unit = UNIMPLEMENTED(0x7DE6711B)
	fun sceRtcSetTick(cpu: CpuState): Unit = UNIMPLEMENTED(0x7ED29E40)
	fun sceRtcIsAlarmed(cpu: CpuState): Unit = UNIMPLEMENTED(0x81FCDA34)
	fun sceRtcCompareTick(cpu: CpuState): Unit = UNIMPLEMENTED(0x9ED0AE87)
	fun sceRtc_A93CF7D8(cpu: CpuState): Unit = UNIMPLEMENTED(0xA93CF7D8)
	fun sceRtc_C2DDBEB5(cpu: CpuState): Unit = UNIMPLEMENTED(0xC2DDBEB5)
	fun sceRtcGetTickResolution(cpu: CpuState): Unit = UNIMPLEMENTED(0xC41C2853)
	fun sceRtcFormatRFC2822(cpu: CpuState): Unit = UNIMPLEMENTED(0xC663B3B9)
	fun sceRtcTickAddWeeks(cpu: CpuState): Unit = UNIMPLEMENTED(0xCF3A2CA8)
	fun sceRtcGetWin32FileTime(cpu: CpuState): Unit = UNIMPLEMENTED(0xCF561893)
	fun sceRtcTickAddMonths(cpu: CpuState): Unit = UNIMPLEMENTED(0xDBF74F1B)
	fun sceRtcParseDateTime(cpu: CpuState): Unit = UNIMPLEMENTED(0xDFBC5F16)
	fun sceRtcGetTime64_t(cpu: CpuState): Unit = UNIMPLEMENTED(0xE1C93E47)
	fun sceRtcTickAddDays(cpu: CpuState): Unit = UNIMPLEMENTED(0xE51B4B7A)
	fun sceRtcTickAddMinutes(cpu: CpuState): Unit = UNIMPLEMENTED(0xE6605BCA)
	fun sceRtcGetCurrentClockLocalTime(cpu: CpuState): Unit = UNIMPLEMENTED(0xE7C27D1B)
	fun sceRtcSetDosTime(cpu: CpuState): Unit = UNIMPLEMENTED(0xF006F264)
	fun sceRtcTickAddSeconds(cpu: CpuState): Unit = UNIMPLEMENTED(0xF2A4AFE5)
	fun sceRtc_F5FCC995(cpu: CpuState): Unit = UNIMPLEMENTED(0xF5FCC995)
	fun sceRtcRegisterCallback(cpu: CpuState): Unit = UNIMPLEMENTED(0xFB3B18CD)

	override fun registerModule() {
		registerFunctionRaw("sceRtcGetAccumulativeTime", 0x011F03C1, since = 150) { sceRtcGetAccumulativeTime(it) }
		registerFunctionRaw("sceRtcGetAccumlativeTime", 0x029CA3B3, since = 150) { sceRtcGetAccumlativeTime(it) }
		registerFunctionRaw("sceRtcFormatRFC3339", 0x0498FB3C, since = 150) { sceRtcFormatRFC3339(it) }
		registerFunctionRaw("sceRtcGetDaysInMonth", 0x05EF322C, since = 150) { sceRtcGetDaysInMonth(it) }
		registerFunctionRaw("sceRtcSetTime64_t", 0x1909C99B, since = 150) { sceRtcSetTime64_t(it) }
		registerFunctionRaw("sceRtcGetLastReincarnatedTime", 0x203CEB0D, since = 150) { sceRtcGetLastReincarnatedTime(it) }
		registerFunctionRaw("sceRtcTickAddMicroseconds", 0x26D25A5D, since = 150) { sceRtcTickAddMicroseconds(it) }
		registerFunctionRaw("sceRtcTickAddHours", 0x26D7A24A, since = 150) { sceRtcTickAddHours(it) }
		registerFunctionRaw("sceRtcGetTime_t", 0x27C4594C, since = 150) { sceRtcGetTime_t(it) }
		registerFunctionRaw("sceRtcFormatRFC3339LocalTime", 0x27F98543, since = 150) { sceRtcFormatRFC3339LocalTime(it) }
		registerFunctionRaw("sceRtcParseRFC3339", 0x28E1E988, since = 150) { sceRtcParseRFC3339(it) }
		registerFunctionRaw("sceRtcConvertUtcToLocalTime", 0x34885E0D, since = 150) { sceRtcConvertUtcToLocalTime(it) }
		registerFunctionRaw("sceRtcGetDosTime", 0x36075567, since = 150) { sceRtcGetDosTime(it) }
		registerFunctionRaw("sceRtcSetTime_t", 0x3A807CC8, since = 150) { sceRtcSetTime_t(it) }
		registerFunctionRaw("sceRtcGetCurrentTick", 0x3F7AD767, since = 150) { sceRtcGetCurrentTick(it) }
		registerFunctionRaw("sceRtcIsLeapYear", 0x42307A17, since = 150) { sceRtcIsLeapYear(it) }
		registerFunctionRaw("sceRtcTickAddYears", 0x42842C77, since = 150) { sceRtcTickAddYears(it) }
		registerFunctionRaw("sceRtcTickAddTicks", 0x44F45E05, since = 150) { sceRtcTickAddTicks(it) }
		registerFunctionRaw("sceRtcCheckValid", 0x4B1B5E82, since = 150) { sceRtcCheckValid(it) }
		registerFunctionRaw("sceRtcGetCurrentClock", 0x4CFA57B0, since = 150) { sceRtcGetCurrentClock(it) }
		registerFunctionRaw("sceRtcGetDayOfWeek", 0x57726BC1, since = 150) { sceRtcGetDayOfWeek(it) }
		registerFunctionRaw("sceRtcGetLastAdjustedTime", 0x62685E98, since = 150) { sceRtcGetLastAdjustedTime(it) }
		registerFunctionRaw("sceRtcUnregisterCallback", 0x6A676D2D, since = 150) { sceRtcUnregisterCallback(it) }
		registerFunctionRaw("sceRtcGetTick", 0x6FF40ACC, since = 150) { sceRtcGetTick(it) }
		registerFunctionRaw("sceRtcConvertLocalTimeToUTC", 0x779242A2, since = 150) { sceRtcConvertLocalTimeToUTC(it) }
		registerFunctionRaw("sceRtcSetWin32FileTime", 0x7ACE4C04, since = 150) { sceRtcSetWin32FileTime(it) }
		registerFunctionRaw("sceRtc_7D1FBED3", 0x7D1FBED3, since = 150) { sceRtc_7D1FBED3(it) }
		registerFunctionRaw("sceRtcFormatRFC2822LocalTime", 0x7DE6711B, since = 150) { sceRtcFormatRFC2822LocalTime(it) }
		registerFunctionRaw("sceRtcSetTick", 0x7ED29E40, since = 150) { sceRtcSetTick(it) }
		registerFunctionRaw("sceRtcIsAlarmed", 0x81FCDA34, since = 150) { sceRtcIsAlarmed(it) }
		registerFunctionRaw("sceRtcCompareTick", 0x9ED0AE87, since = 150) { sceRtcCompareTick(it) }
		registerFunctionRaw("sceRtc_A93CF7D8", 0xA93CF7D8, since = 150) { sceRtc_A93CF7D8(it) }
		registerFunctionRaw("sceRtc_C2DDBEB5", 0xC2DDBEB5, since = 150) { sceRtc_C2DDBEB5(it) }
		registerFunctionRaw("sceRtcGetTickResolution", 0xC41C2853, since = 150) { sceRtcGetTickResolution(it) }
		registerFunctionRaw("sceRtcFormatRFC2822", 0xC663B3B9, since = 150) { sceRtcFormatRFC2822(it) }
		registerFunctionRaw("sceRtcTickAddWeeks", 0xCF3A2CA8, since = 150) { sceRtcTickAddWeeks(it) }
		registerFunctionRaw("sceRtcGetWin32FileTime", 0xCF561893, since = 150) { sceRtcGetWin32FileTime(it) }
		registerFunctionRaw("sceRtcTickAddMonths", 0xDBF74F1B, since = 150) { sceRtcTickAddMonths(it) }
		registerFunctionRaw("sceRtcParseDateTime", 0xDFBC5F16, since = 150) { sceRtcParseDateTime(it) }
		registerFunctionRaw("sceRtcGetTime64_t", 0xE1C93E47, since = 150) { sceRtcGetTime64_t(it) }
		registerFunctionRaw("sceRtcTickAddDays", 0xE51B4B7A, since = 150) { sceRtcTickAddDays(it) }
		registerFunctionRaw("sceRtcTickAddMinutes", 0xE6605BCA, since = 150) { sceRtcTickAddMinutes(it) }
		registerFunctionRaw("sceRtcGetCurrentClockLocalTime", 0xE7C27D1B, since = 150) { sceRtcGetCurrentClockLocalTime(it) }
		registerFunctionRaw("sceRtcSetDosTime", 0xF006F264, since = 150) { sceRtcSetDosTime(it) }
		registerFunctionRaw("sceRtcTickAddSeconds", 0xF2A4AFE5, since = 150) { sceRtcTickAddSeconds(it) }
		registerFunctionRaw("sceRtc_F5FCC995", 0xF5FCC995, since = 150) { sceRtc_F5FCC995(it) }
		registerFunctionRaw("sceRtcRegisterCallback", 0xFB3B18CD, since = 150) { sceRtcRegisterCallback(it) }
	}
}
