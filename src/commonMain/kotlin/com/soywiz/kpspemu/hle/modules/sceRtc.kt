package com.soywiz.kpspemu.hle.modules

import com.soywiz.klock.*
import com.soywiz.kmem.*
import com.soywiz.kpspemu.*
import com.soywiz.kpspemu.cpu.*
import com.soywiz.kpspemu.hle.*
import com.soywiz.kpspemu.hle.manager.*
import com.soywiz.kpspemu.mem.*
import com.soywiz.kmem.clamp

@Suppress("UNUSED_PARAMETER")
class sceRtc(emulator: Emulator) : SceModule(emulator, "sceRtc", 0x40010011, "rtc.prx", "sceRTC_Service") {
    fun sceRtcGetCurrentTick(ptr: Ptr): Int = 0.apply { ptr.sdw(0, timeManager.getTimeInMicroseconds()) }
    fun sceRtcGetTickResolution(): Int = 1000000
    fun sceRtcGetDayOfWeek(year: Int, month: Int, day: Int): Int = DateTime(year, month, day).dayOfWeekInt
    fun sceRtcGetDaysInMonth(year: Int, month: Int): Int = YearMonth(year, month).days
    fun sceRtcSetTick(datePtr: PtrStruct<ScePspDateTime>, ticksPtr: Ptr64): Int {
        datePtr.set(ScePspDateTime(ticksPtr.get()))
        return 0
    }

    fun sceRtcGetTick(datePtr: PtrStruct<ScePspDateTime>, ticksPtr: Ptr64): Int {
        ticksPtr.set(datePtr.get().tick)
        return 0
    }

    fun sceRtcGetCurrentClock(date: PtrStruct<ScePspDateTime>, timezone: Int): Int {
        date.set(ScePspDateTime(DateTime.now().toOffset(timezone.clamp(-600000, +600000).milliseconds)))
        return 0
    }

    fun sceRtcGetCurrentClockLocalTime(date: PtrStruct<ScePspDateTime>): Int {
        date.set(ScePspDateTime(DateTime.now().local))
        return 0
    }

    fun sceRtcTickAddTicks(dst: Ptr64, src: Ptr64, count: Long): Int {
        dst.set(src.get() + count)
        return 0
    }

    fun sceRtcTickAddMicroseconds(dst: Ptr64, src: Ptr64, count: Long): Int {
        return sceRtcTickAddTicks(dst, src, count)
    }

    fun sceRtcTickAddSeconds(dst: Ptr64, src: Ptr64, count: Long): Int {
        return sceRtcTickAddMicroseconds(dst, src, count * 1_000_000L)
    }

    fun sceRtcTickAddMinutes(dst: Ptr64, src: Ptr64, count: Long): Int {
        return sceRtcTickAddSeconds(dst, src, count * 60)
    }

    fun sceRtcTickAddHours(dst: Ptr64, src: Ptr64, count: Int): Int {
        return sceRtcTickAddSeconds(dst, src, count.toLong() * 3600)
    }

    private fun _sceRtcTickAddTimeDist(dst: Ptr64, src: Ptr64, distance: TimeDistance): Int {
        val srcDate = ScePspDateTime(src.get())
        val newDate = srcDate.date + distance
        dst.set(ScePspDateTime(newDate.local, srcDate.microAdjust).tick)
        return 0
    }

    fun sceRtcTickAddDays(dst: Ptr64, src: Ptr64, count: Int): Int = _sceRtcTickAddTimeDist(dst, src, 0.months + count.days)
    fun sceRtcTickAddWeeks(dst: Ptr64, src: Ptr64, count: Int): Int = _sceRtcTickAddTimeDist(dst, src, 0.months + (count * 7).days)
    fun sceRtcTickAddMonths(dst: Ptr64, src: Ptr64, count: Int): Int = _sceRtcTickAddTimeDist(dst, src, count.months + 0.days)
    fun sceRtcTickAddYears(dst: Ptr64, src: Ptr64, count: Int): Int = _sceRtcTickAddTimeDist(dst, src, count.years + 0.days)

    fun sceRtcCompareTick(tick1: Ptr64, tick2: Ptr64): Int {
        return tick1.get().compareTo(tick2.get())
    }

    fun sceRtcGetAccumulativeTime(cpu: CpuState): Unit = UNIMPLEMENTED(0x011F03C1)
    fun sceRtcGetAccumlativeTime(cpu: CpuState): Unit = UNIMPLEMENTED(0x029CA3B3)
    fun sceRtcFormatRFC3339(cpu: CpuState): Unit = UNIMPLEMENTED(0x0498FB3C)
    fun sceRtcSetTime64_t(cpu: CpuState): Unit = UNIMPLEMENTED(0x1909C99B)
    fun sceRtcGetLastReincarnatedTime(cpu: CpuState): Unit = UNIMPLEMENTED(0x203CEB0D)
    fun sceRtcGetTime_t(cpu: CpuState): Unit = UNIMPLEMENTED(0x27C4594C)
    fun sceRtcFormatRFC3339LocalTime(cpu: CpuState): Unit = UNIMPLEMENTED(0x27F98543)
    fun sceRtcParseRFC3339(cpu: CpuState): Unit = UNIMPLEMENTED(0x28E1E988)
    fun sceRtcConvertUtcToLocalTime(cpu: CpuState): Unit = UNIMPLEMENTED(0x34885E0D)
    fun sceRtcGetDosTime(cpu: CpuState): Unit = UNIMPLEMENTED(0x36075567)
    fun sceRtcSetTime_t(cpu: CpuState): Unit = UNIMPLEMENTED(0x3A807CC8)
    fun sceRtcIsLeapYear(cpu: CpuState): Unit = UNIMPLEMENTED(0x42307A17)
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
    fun sceRtcGetWin32FileTime(cpu: CpuState): Unit = UNIMPLEMENTED(0xCF561893)
    fun sceRtcParseDateTime(cpu: CpuState): Unit = UNIMPLEMENTED(0xDFBC5F16)
    fun sceRtcGetTime64_t(cpu: CpuState): Unit = UNIMPLEMENTED(0xE1C93E47)
    fun sceRtcSetDosTime(cpu: CpuState): Unit = UNIMPLEMENTED(0xF006F264)
    fun sceRtc_F5FCC995(cpu: CpuState): Unit = UNIMPLEMENTED(0xF5FCC995)
    fun sceRtcRegisterCallback(cpu: CpuState): Unit = UNIMPLEMENTED(0xFB3B18CD)

    override fun registerModule() {
        registerFunctionInt("sceRtcGetCurrentTick", 0x3F7AD767, since = 150) { sceRtcGetCurrentTick(ptr) }
        registerFunctionInt("sceRtcGetTickResolution", 0xC41C2853, since = 150) { sceRtcGetTickResolution() }
        registerFunctionInt("sceRtcGetDayOfWeek", 0x57726BC1, since = 150) { sceRtcGetDayOfWeek(int, int, int) }
        registerFunctionInt("sceRtcGetDaysInMonth", 0x05EF322C, since = 150) { sceRtcGetDaysInMonth(int, int) }
        registerFunctionInt("sceRtcSetTick", 0x7ED29E40, since = 150) { sceRtcSetTick(ptr(ScePspDateTime), ptr64) }
        registerFunctionInt("sceRtcGetTick", 0x6FF40ACC, since = 150) { sceRtcGetTick(ptr(ScePspDateTime), ptr64) }
        registerFunctionInt(
            "sceRtcGetCurrentClock",
            0x4CFA57B0,
            since = 150
        ) { sceRtcGetCurrentClock(ptr(ScePspDateTime), int) }
        registerFunctionInt("sceRtcGetCurrentClockLocalTime", 0xE7C27D1B, since = 150) {
            sceRtcGetCurrentClockLocalTime(
                ptr(ScePspDateTime)
            )
        }
        registerFunctionInt("sceRtcCompareTick", 0x9ED0AE87, since = 150) { sceRtcCompareTick(ptr64, ptr64) }
        registerFunctionInt("sceRtcTickAddTicks", 0x44F45E05, since = 150) { sceRtcTickAddTicks(ptr64, ptr64, long) }
        registerFunctionInt("sceRtcTickAddMicroseconds", 0x26D25A5D, since = 150) {
            sceRtcTickAddMicroseconds(
                ptr64,
                ptr64,
                long
            )
        }
        registerFunctionInt("sceRtcTickAddSeconds", 0xF2A4AFE5, since = 150) {
            sceRtcTickAddSeconds(
                ptr64,
                ptr64,
                long
            )
        }
        registerFunctionInt("sceRtcTickAddMinutes", 0xE6605BCA, since = 150) {
            sceRtcTickAddMinutes(
                ptr64,
                ptr64,
                long
            )
        }
        registerFunctionInt("sceRtcTickAddHours", 0x26D7A24A, since = 150) { sceRtcTickAddHours(ptr64, ptr64, int) }
        registerFunctionInt("sceRtcTickAddDays", 0xE51B4B7A, since = 150) { sceRtcTickAddDays(ptr64, ptr64, int) }
        registerFunctionInt("sceRtcTickAddWeeks", 0xCF3A2CA8, since = 150) { sceRtcTickAddWeeks(ptr64, ptr64, int) }
        registerFunctionInt("sceRtcTickAddMonths", 0xDBF74F1B, since = 150) { sceRtcTickAddMonths(ptr64, ptr64, int) }
        registerFunctionInt("sceRtcTickAddYears", 0x42842C77, since = 150) { sceRtcTickAddYears(ptr64, ptr64, int) }

        registerFunctionRaw("sceRtcGetAccumulativeTime", 0x011F03C1, since = 150) { sceRtcGetAccumulativeTime(it) }
        registerFunctionRaw("sceRtcGetAccumlativeTime", 0x029CA3B3, since = 150) { sceRtcGetAccumlativeTime(it) }
        registerFunctionRaw("sceRtcFormatRFC3339", 0x0498FB3C, since = 150) { sceRtcFormatRFC3339(it) }
        registerFunctionRaw("sceRtcSetTime64_t", 0x1909C99B, since = 150) { sceRtcSetTime64_t(it) }
        registerFunctionRaw(
            "sceRtcGetLastReincarnatedTime",
            0x203CEB0D,
            since = 150
        ) { sceRtcGetLastReincarnatedTime(it) }
        registerFunctionRaw("sceRtcGetTime_t", 0x27C4594C, since = 150) { sceRtcGetTime_t(it) }
        registerFunctionRaw(
            "sceRtcFormatRFC3339LocalTime",
            0x27F98543,
            since = 150
        ) { sceRtcFormatRFC3339LocalTime(it) }
        registerFunctionRaw("sceRtcParseRFC3339", 0x28E1E988, since = 150) { sceRtcParseRFC3339(it) }
        registerFunctionRaw("sceRtcConvertUtcToLocalTime", 0x34885E0D, since = 150) { sceRtcConvertUtcToLocalTime(it) }
        registerFunctionRaw("sceRtcGetDosTime", 0x36075567, since = 150) { sceRtcGetDosTime(it) }
        registerFunctionRaw("sceRtcSetTime_t", 0x3A807CC8, since = 150) { sceRtcSetTime_t(it) }
        registerFunctionRaw("sceRtcIsLeapYear", 0x42307A17, since = 150) { sceRtcIsLeapYear(it) }
        registerFunctionRaw("sceRtcCheckValid", 0x4B1B5E82, since = 150) { sceRtcCheckValid(it) }
        registerFunctionRaw("sceRtcGetLastAdjustedTime", 0x62685E98, since = 150) { sceRtcGetLastAdjustedTime(it) }
        registerFunctionRaw("sceRtcUnregisterCallback", 0x6A676D2D, since = 150) { sceRtcUnregisterCallback(it) }
        registerFunctionRaw("sceRtcConvertLocalTimeToUTC", 0x779242A2, since = 150) { sceRtcConvertLocalTimeToUTC(it) }
        registerFunctionRaw("sceRtcSetWin32FileTime", 0x7ACE4C04, since = 150) { sceRtcSetWin32FileTime(it) }
        registerFunctionRaw("sceRtc_7D1FBED3", 0x7D1FBED3, since = 150) { sceRtc_7D1FBED3(it) }
        registerFunctionRaw(
            "sceRtcFormatRFC2822LocalTime",
            0x7DE6711B,
            since = 150
        ) { sceRtcFormatRFC2822LocalTime(it) }
        registerFunctionRaw("sceRtcIsAlarmed", 0x81FCDA34, since = 150) { sceRtcIsAlarmed(it) }
        registerFunctionRaw("sceRtc_A93CF7D8", 0xA93CF7D8, since = 150) { sceRtc_A93CF7D8(it) }
        registerFunctionRaw("sceRtc_C2DDBEB5", 0xC2DDBEB5, since = 150) { sceRtc_C2DDBEB5(it) }
        registerFunctionRaw("sceRtcFormatRFC2822", 0xC663B3B9, since = 150) { sceRtcFormatRFC2822(it) }
        registerFunctionRaw("sceRtcGetWin32FileTime", 0xCF561893, since = 150) { sceRtcGetWin32FileTime(it) }
        registerFunctionRaw("sceRtcParseDateTime", 0xDFBC5F16, since = 150) { sceRtcParseDateTime(it) }
        registerFunctionRaw("sceRtcGetTime64_t", 0xE1C93E47, since = 150) { sceRtcGetTime64_t(it) }
        registerFunctionRaw("sceRtcSetDosTime", 0xF006F264, since = 150) { sceRtcSetDosTime(it) }
        registerFunctionRaw("sceRtc_F5FCC995", 0xF5FCC995, since = 150) { sceRtc_F5FCC995(it) }
        registerFunctionRaw("sceRtcRegisterCallback", 0xFB3B18CD, since = 150) { sceRtcRegisterCallback(it) }
    }
}
