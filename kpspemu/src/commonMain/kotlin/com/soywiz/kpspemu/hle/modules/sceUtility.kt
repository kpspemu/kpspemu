package com.soywiz.kpspemu.hle.modules

import com.soywiz.korio.i18n.*
import com.soywiz.korio.util.*
import com.soywiz.kpspemu.*
import com.soywiz.kpspemu.cpu.*
import com.soywiz.kpspemu.hle.*
import com.soywiz.kpspemu.hle.error.*
import com.soywiz.kpspemu.mem.*
import com.soywiz.kpspemu.util.*

@Suppress("UNUSED_PARAMETER", "MemberVisibilityCanPrivate", "FunctionName")
class sceUtility(emulator: Emulator) :
    SceModule(emulator, "sceUtility", 0x40010011, "utility.prx", "sceUtility_Driver") {
    fun sceUtilitySavedataInitStart(paramsPtr: Ptr): Int {
        logger.error { "sceUtilitySavedataInitStart: $paramsPtr" }
        paramsPtr.capture(SceUtilitySavedataParam) { params ->
            params.base.result = SceKernelErrors.ERROR_SAVEDATA_LOAD_NO_DATA
        }
        //return Promise2.resolve(this._sceUtilitySavedataInitStart(paramsPtr.clone())).then(result => {
        //	var params = SceUtilitySavedataParam.struct.read(paramsPtr.clone());
        //	params.base.result = result;
        //	return 0;
        //});
        return 0
    }

    fun sceUtilitySavedataGetStatus(): Int {
        logger.error { "sceUtilitySavedataGetStatus" }
        return 0
    }

    companion object {
        const val PSP_SYSTEMPARAM_ID_STRING_NICKNAME = 1
        const val PSP_SYSTEMPARAM_ID_INT_ADHOC_CHANNEL = 2
        const val PSP_SYSTEMPARAM_ID_INT_WLAN_POWERSAVE = 3
        const val PSP_SYSTEMPARAM_ID_INT_DATE_FORMAT = 4
        const val PSP_SYSTEMPARAM_ID_INT_TIME_FORMAT = 5
        const val PSP_SYSTEMPARAM_ID_INT_TIMEZONE = 6
        const val PSP_SYSTEMPARAM_ID_INT_DAYLIGHTSAVINGS = 7
        const val PSP_SYSTEMPARAM_ID_INT_LANGUAGE = 8
        const val PSP_SYSTEMPARAM_ID_INT_UNKNOWN = 9
        const val PSP_SYSTEMPARAM_RETVAL_OK = 0
        const val PSP_SYSTEMPARAM_RETVAL_FAIL = 0x80110103
        const val PSP_SYSTEMPARAM_ADHOC_CHANNEL_AUTOMATIC = 0
        const val PSP_SYSTEMPARAM_ADHOC_CHANNEL_1 = 1
        const val PSP_SYSTEMPARAM_ADHOC_CHANNEL_6 = 6
        const val PSP_SYSTEMPARAM_ADHOC_CHANNEL_11 = 11
        const val PSP_SYSTEMPARAM_WLAN_POWERSAVE_OFF = 0
        const val PSP_SYSTEMPARAM_WLAN_POWERSAVE_ON = 1
        const val PSP_SYSTEMPARAM_DATE_FORMAT_YYYYMMDD = 0
        const val PSP_SYSTEMPARAM_DATE_FORMAT_MMDDYYYY = 1
        const val PSP_SYSTEMPARAM_DATE_FORMAT_DDMMYYYY = 2
        const val PSP_SYSTEMPARAM_TIME_FORMAT_24HR = 0
        const val PSP_SYSTEMPARAM_TIME_FORMAT_12HR = 1
        const val PSP_SYSTEMPARAM_DAYLIGHTSAVINGS_STD = 0
        const val PSP_SYSTEMPARAM_DAYLIGHTSAVINGS_SAVING = 1
        const val PSP_SYSTEMPARAM_LANGUAGE_JAPANESE = 0
        const val PSP_SYSTEMPARAM_LANGUAGE_ENGLISH = 1
        const val PSP_SYSTEMPARAM_LANGUAGE_FRENCH = 2
        const val PSP_SYSTEMPARAM_LANGUAGE_SPANISH = 3
        const val PSP_SYSTEMPARAM_LANGUAGE_GERMAN = 4
        const val PSP_SYSTEMPARAM_LANGUAGE_ITALIAN = 5
        const val PSP_SYSTEMPARAM_LANGUAGE_DUTCH = 6
        const val PSP_SYSTEMPARAM_LANGUAGE_PORTUGUESE = 7
        const val PSP_SYSTEMPARAM_LANGUAGE_RUSSIAN = 8
        const val PSP_SYSTEMPARAM_LANGUAGE_KOREAN = 9
        const val PSP_SYSTEMPARAM_LANGUAGE_CHINESE_TRADITIONAL = 10
        const val PSP_SYSTEMPARAM_LANGUAGE_CHINESE_SIMPLIFIED = 11
    }

    // @TODO: Move to a system configuration class and make it mutable
    val adhocChannel = 0
    val wlanPowersave = 0
    val dateFormat = PSP_SYSTEMPARAM_DATE_FORMAT_YYYYMMDD
    val timeFormat = PSP_SYSTEMPARAM_TIME_FORMAT_24HR
    val language get() = Language.CURRENT.pspLanguage
    val timezone = 0
    val daylightSavings = PSP_SYSTEMPARAM_DAYLIGHTSAVINGS_STD

    val Language.pspLanguage: Int get() = when (this) {
        Language.ENGLISH -> PSP_SYSTEMPARAM_LANGUAGE_ENGLISH
        Language.JAPANESE -> PSP_SYSTEMPARAM_LANGUAGE_JAPANESE
        Language.KOREAN -> PSP_SYSTEMPARAM_LANGUAGE_KOREAN
        Language.CHINESE -> PSP_SYSTEMPARAM_LANGUAGE_CHINESE_SIMPLIFIED
        Language.SPANISH -> PSP_SYSTEMPARAM_LANGUAGE_SPANISH
        Language.ITALIAN -> PSP_SYSTEMPARAM_LANGUAGE_ITALIAN
        Language.FRENCH -> PSP_SYSTEMPARAM_LANGUAGE_FRENCH
        Language.GERMAN -> PSP_SYSTEMPARAM_LANGUAGE_GERMAN
        Language.DUTCH -> PSP_SYSTEMPARAM_LANGUAGE_DUTCH
        Language.RUSSIAN -> PSP_SYSTEMPARAM_LANGUAGE_RUSSIAN
        Language.PORTUGUESE -> PSP_SYSTEMPARAM_LANGUAGE_PORTUGUESE
        else -> PSP_SYSTEMPARAM_LANGUAGE_ENGLISH
    }

    fun sceUtilityGetSystemParamInt(id: Int, value: Ptr): Int {
        logger.trace { "Not implemented: sceUtilityGetSystemParamInt:$id,$value" }
        val avalue = when (id) {
            PSP_SYSTEMPARAM_ID_STRING_NICKNAME -> -1
            PSP_SYSTEMPARAM_ID_INT_ADHOC_CHANNEL -> adhocChannel
            PSP_SYSTEMPARAM_ID_INT_WLAN_POWERSAVE -> wlanPowersave
            PSP_SYSTEMPARAM_ID_INT_DATE_FORMAT -> dateFormat
            PSP_SYSTEMPARAM_ID_INT_TIME_FORMAT -> timeFormat
            PSP_SYSTEMPARAM_ID_INT_TIMEZONE -> timezone
            PSP_SYSTEMPARAM_ID_INT_DAYLIGHTSAVINGS -> daylightSavings
            PSP_SYSTEMPARAM_ID_INT_LANGUAGE -> language
            PSP_SYSTEMPARAM_ID_INT_UNKNOWN -> -1
            else -> -1
        }
        value.sw(0, avalue)
        return PSP_SYSTEMPARAM_RETVAL_OK
    }

    var currentStep = DialogStepEnum.NONE

    fun sceUtilityMsgDialogInitStart(paramsPtr: Ptr): Int {
        logger.error { "sceUtilityMsgDialogInitStart:$paramsPtr" }
        paramsPtr.capture(PspUtilityMsgDialogParams) { params ->
            params.buttonPressed = PspUtilityMsgDialogPressed.PSP_UTILITY_MSGDIALOG_RESULT_YES
            this.currentStep = DialogStepEnum.SUCCESS
        }
        return 0
    }

    fun sceUtilityMsgDialogGetStatus(): Int {
        logger.error { "sceUtilityMsgDialogGetStatus" }
        try {
            return this.currentStep.id
        } finally {
            if (this.currentStep === DialogStepEnum.SHUTDOWN) this.currentStep = DialogStepEnum.NONE
        }
    }

    fun sceUtilityMsgDialogShutdownStart(): Unit {
        logger.error { "sceUtilityMsgDialogShutdownStart" }
        currentStep = DialogStepEnum.SHUTDOWN
    }

    fun sceUtilityLoadAvModule(module: Int): Int {
        logger.error { "sceUtilityLoadAvModule" }
        //getModule<LoadExecForUser>()
        return 1
    }

    fun sceUtility_0251B134(cpu: CpuState): Unit = UNIMPLEMENTED(0x0251B134)
    fun sceUtilityHtmlViewerUpdate(cpu: CpuState): Unit = UNIMPLEMENTED(0x05AFB9E4)
    fun sceUtility_06A48659(cpu: CpuState): Unit = UNIMPLEMENTED(0x06A48659)
    fun sceUtilityLoadUsbModule(cpu: CpuState): Unit = UNIMPLEMENTED(0x0D5BC6D2)
    fun sceUtility_0F3EEAAC(cpu: CpuState): Unit = UNIMPLEMENTED(0x0F3EEAAC)
    fun sceUtilityInstallInitStart(cpu: CpuState): Unit = UNIMPLEMENTED(0x1281DA8E)
    fun sceUtility_147F7C85(cpu: CpuState): Unit = UNIMPLEMENTED(0x147F7C85)
    fun sceUtility_149A7895(cpu: CpuState): Unit = UNIMPLEMENTED(0x149A7895)
    fun sceUtilityLoadNetModule(cpu: CpuState): Unit = UNIMPLEMENTED(0x1579A159)
    fun sceUtility_16A1A8D8(cpu: CpuState): Unit = UNIMPLEMENTED(0x16A1A8D8)
    fun sceUtility_16D02AF0(cpu: CpuState): Unit = UNIMPLEMENTED(0x16D02AF0)
    fun sceUtility_28D35634(cpu: CpuState): Unit = UNIMPLEMENTED(0x28D35634)
    fun sceUtility_2995D020(cpu: CpuState): Unit = UNIMPLEMENTED(0x2995D020)
    fun sceUtilityLoadModule(cpu: CpuState): Unit = UNIMPLEMENTED(0x2A2B3DE0)
    fun sceUtility_2B96173B(cpu: CpuState): Unit = UNIMPLEMENTED(0x2B96173B)

    fun sceUtilityGetSystemParamString(id: Int, strPtr: Ptr, len: Int): Int {
        if (id == PSP_SYSTEMPARAM_ID_STRING_NICKNAME) {
            strPtr.writeStringz("username")
            return 0
        }
        return -1
    }

    fun sceUtility_3AAD51DC(cpu: CpuState): Unit = UNIMPLEMENTED(0x3AAD51DC)
    fun sceNetplayDialogInitStart(cpu: CpuState): Unit = UNIMPLEMENTED(0x3AD50AE7)
    fun sceUtilityOskShutdownStart(cpu: CpuState): Unit = UNIMPLEMENTED(0x3DFAEBA9)
    fun sceNetplayDialogUpdate(cpu: CpuState): Unit = UNIMPLEMENTED(0x417BED54)
    fun sceUtilitySetSystemParamString(cpu: CpuState): Unit = UNIMPLEMENTED(0x41E30674)
    fun sceUtility_42071A83(cpu: CpuState): Unit = UNIMPLEMENTED(0x42071A83)
    fun sceUtilityGetNetParam(cpu: CpuState): Unit = UNIMPLEMENTED(0x434D4B3A)
    fun sceUtilitySetSystemParamInt(cpu: CpuState): Unit = UNIMPLEMENTED(0x45C18506)
    fun sceUtilityMsgDialogAbort(cpu: CpuState): Unit = UNIMPLEMENTED(0x4928BD96)
    fun sceUtility_4A833BA4(cpu: CpuState): Unit = UNIMPLEMENTED(0x4A833BA4)
    fun sceUtility_4B0A8FE5(cpu: CpuState): Unit = UNIMPLEMENTED(0x4B0A8FE5)
    fun sceUtilityOskUpdate(cpu: CpuState): Unit = UNIMPLEMENTED(0x4B85C861)
    fun sceUtilityNetconfInitStart(cpu: CpuState): Unit = UNIMPLEMENTED(0x4DB1E739)
    fun sceUtilityGetNetParamLatestID(cpu: CpuState): Unit = UNIMPLEMENTED(0x4FED24D8)
    fun sceUtility_54A5C62F(cpu: CpuState): Unit = UNIMPLEMENTED(0x54A5C62F)
    fun sceUtilityCheckNetParam(cpu: CpuState): Unit = UNIMPLEMENTED(0x5EEE6548)
    fun sceUtilityInstallShutdownStart(cpu: CpuState): Unit = UNIMPLEMENTED(0x5EF1C24A)
    fun sceUtilityNetconfGetStatus(cpu: CpuState): Unit = UNIMPLEMENTED(0x6332AA39)
    fun sceUtilityUnloadNetModule(cpu: CpuState): Unit = UNIMPLEMENTED(0x64D50C56)
    fun sceUtility_6F56F9CF(cpu: CpuState): Unit = UNIMPLEMENTED(0x6F56F9CF)
    fun sceUtility_70267ADF(cpu: CpuState): Unit = UNIMPLEMENTED(0x70267ADF)
    fun sceUtilityGameSharingUpdate(cpu: CpuState): Unit = UNIMPLEMENTED(0x7853182D)
    fun sceUtility_81C44706(cpu: CpuState): Unit = UNIMPLEMENTED(0x81C44706)
    fun sceUtility_8326AB05(cpu: CpuState): Unit = UNIMPLEMENTED(0x8326AB05)
    fun sceUtility_86A03A27(cpu: CpuState): Unit = UNIMPLEMENTED(0x86A03A27)
    fun sceUtility_86ABDB1B(cpu: CpuState): Unit = UNIMPLEMENTED(0x86ABDB1B)
    fun sceUtility_88BC7406(cpu: CpuState): Unit = UNIMPLEMENTED(0x88BC7406)
    fun sceUtility_89317C8F(cpu: CpuState): Unit = UNIMPLEMENTED(0x89317C8F)
    fun sceUtilityNetconfUpdate(cpu: CpuState): Unit = UNIMPLEMENTED(0x91E70E35)
    fun sceUtility_943CBA46(cpu: CpuState): Unit = UNIMPLEMENTED(0x943CBA46)
    fun sceUtilityGameSharingGetStatus(cpu: CpuState): Unit = UNIMPLEMENTED(0x946963F3)
    fun sceUtilityMsgDialogUpdate(cpu: CpuState): Unit = UNIMPLEMENTED(0x95FC253B)
    fun sceUtilitySavedataShutdownStart(cpu: CpuState): Unit = UNIMPLEMENTED(0x9790B33C)
    fun sceUtilityInstallUpdate(cpu: CpuState): Unit = UNIMPLEMENTED(0xA03D29BA)
    fun sceUtility_A084E056(cpu: CpuState): Unit = UNIMPLEMENTED(0xA084E056)
    fun sceUtility_A50E5B30(cpu: CpuState): Unit = UNIMPLEMENTED(0xA50E5B30)
    fun sceUtility_AB083EA9(cpu: CpuState): Unit = UNIMPLEMENTED(0xAB083EA9)
    fun sceUtility_B0FB7FF5(cpu: CpuState): Unit = UNIMPLEMENTED(0xB0FB7FF5)
    fun sceUtility_B62A4061(cpu: CpuState): Unit = UNIMPLEMENTED(0xB62A4061)
    fun sceNetplayDialogGetStatus(cpu: CpuState): Unit = UNIMPLEMENTED(0xB6CEE597)
    fun sceUtility_B8592D5F(cpu: CpuState): Unit = UNIMPLEMENTED(0xB8592D5F)
    fun sceNetplayDialogShutdownStart(cpu: CpuState): Unit = UNIMPLEMENTED(0xBC6B6296)
    fun sceUtilityHtmlViewerGetStatus(cpu: CpuState): Unit = UNIMPLEMENTED(0xBDA7D894)
    fun sceUtilityInstallGetStatus(cpu: CpuState): Unit = UNIMPLEMENTED(0xC4700FA3)
    fun sceUtilityGameSharingInitStart(cpu: CpuState): Unit = UNIMPLEMENTED(0xC492F751)
    fun sceUtilityHtmlViewerInitStart(cpu: CpuState): Unit = UNIMPLEMENTED(0xCDC3AA41)
    fun sceUtility_D17A0573(cpu: CpuState): Unit = UNIMPLEMENTED(0xD17A0573)
    fun sceUtilitySavedataUpdate(cpu: CpuState): Unit = UNIMPLEMENTED(0xD4B95FFB)
    fun sceUtility_D81957B7(cpu: CpuState): Unit = UNIMPLEMENTED(0xD81957B7)
    fun sceUtility_D852CDCE(cpu: CpuState): Unit = UNIMPLEMENTED(0xD852CDCE)
    fun sceUtility_DA97F1AA(cpu: CpuState): Unit = UNIMPLEMENTED(0xDA97F1AA)
    fun sceUtility_DDE5389D(cpu: CpuState): Unit = UNIMPLEMENTED(0xDDE5389D)
    fun sceUtility_E19C97D6(cpu: CpuState): Unit = UNIMPLEMENTED(0xE19C97D6)
    fun sceUtilityUnloadModule(cpu: CpuState): Unit = UNIMPLEMENTED(0xE49BFE92)
    fun sceUtility_E7B778D8(cpu: CpuState): Unit = UNIMPLEMENTED(0xE7B778D8)
    fun sceUtility_ECE1D3E5(cpu: CpuState): Unit = UNIMPLEMENTED(0xECE1D3E5)
    fun sceUtility_ED0FAD38(cpu: CpuState): Unit = UNIMPLEMENTED(0xED0FAD38)
    fun sceUtility_EF3582B2(cpu: CpuState): Unit = UNIMPLEMENTED(0xEF3582B2)
    fun sceUtilityGameSharingShutdownStart(cpu: CpuState): Unit = UNIMPLEMENTED(0xEFC6F80F)
    fun sceUtilityOskGetStatus(cpu: CpuState): Unit = UNIMPLEMENTED(0xF3F76017)
    fun sceUtility_F3FBC572(cpu: CpuState): Unit = UNIMPLEMENTED(0xF3FBC572)
    fun sceUtilityHtmlViewerShutdownStart(cpu: CpuState): Unit = UNIMPLEMENTED(0xF5CE1134)
    fun sceUtilityOskInitStart(cpu: CpuState): Unit = UNIMPLEMENTED(0xF6269B82)
    fun sceUtilityUnloadUsbModule(cpu: CpuState): Unit = UNIMPLEMENTED(0xF64910F0)
    fun sceUtilityUnloadAvModule(cpu: CpuState): Unit = UNIMPLEMENTED(0xF7D8D092)
    fun sceUtilityNetconfShutdownStart(cpu: CpuState): Unit = UNIMPLEMENTED(0xF88155F6)
    fun sceUtility_F9E0008C(cpu: CpuState): Unit = UNIMPLEMENTED(0xF9E0008C)


    override fun registerModule() {
        registerFunctionInt("sceUtilitySavedataInitStart", 0x50C4CD57, since = 150) { sceUtilitySavedataInitStart(ptr) }
        registerFunctionInt("sceUtilitySavedataGetStatus", 0x8874DBE0, since = 150) { sceUtilitySavedataGetStatus() }
        registerFunctionInt("sceUtilityGetSystemParamInt", 0xA5DA2406, since = 150) {
            sceUtilityGetSystemParamInt(
                int,
                ptr
            )
        }
        registerFunctionInt(
            "sceUtilityMsgDialogInitStart",
            0x2AD8E239,
            since = 150
        ) { sceUtilityMsgDialogInitStart(ptr) }
        registerFunctionInt("sceUtilityMsgDialogGetStatus", 0x9A1C91D7, since = 150) { sceUtilityMsgDialogGetStatus() }
        registerFunctionVoid(
            "sceUtilityMsgDialogShutdownStart",
            0x67AF3428,
            since = 150
        ) { sceUtilityMsgDialogShutdownStart() }
        registerFunctionInt("sceUtilityLoadAvModule", 0xC629AF26, since = 150) { sceUtilityLoadAvModule(int) }

        registerFunctionRaw("sceUtility_0251B134", 0x0251B134, since = 150) { sceUtility_0251B134(it) }
        registerFunctionRaw("sceUtilityHtmlViewerUpdate", 0x05AFB9E4, since = 150) { sceUtilityHtmlViewerUpdate(it) }
        registerFunctionRaw("sceUtility_06A48659", 0x06A48659, since = 150) { sceUtility_06A48659(it) }
        registerFunctionRaw("sceUtilityLoadUsbModule", 0x0D5BC6D2, since = 150) { sceUtilityLoadUsbModule(it) }
        registerFunctionRaw("sceUtility_0F3EEAAC", 0x0F3EEAAC, since = 150) { sceUtility_0F3EEAAC(it) }
        registerFunctionRaw("sceUtilityInstallInitStart", 0x1281DA8E, since = 150) { sceUtilityInstallInitStart(it) }
        registerFunctionRaw("sceUtility_147F7C85", 0x147F7C85, since = 150) { sceUtility_147F7C85(it) }
        registerFunctionRaw("sceUtility_149A7895", 0x149A7895, since = 150) { sceUtility_149A7895(it) }
        registerFunctionRaw("sceUtilityLoadNetModule", 0x1579A159, since = 150) { sceUtilityLoadNetModule(it) }
        registerFunctionRaw("sceUtility_16A1A8D8", 0x16A1A8D8, since = 150) { sceUtility_16A1A8D8(it) }
        registerFunctionRaw("sceUtility_16D02AF0", 0x16D02AF0, since = 150) { sceUtility_16D02AF0(it) }
        registerFunctionRaw("sceUtility_28D35634", 0x28D35634, since = 150) { sceUtility_28D35634(it) }
        registerFunctionRaw("sceUtility_2995D020", 0x2995D020, since = 150) { sceUtility_2995D020(it) }
        registerFunctionRaw("sceUtilityLoadModule", 0x2A2B3DE0, since = 150) { sceUtilityLoadModule(it) }
        registerFunctionRaw("sceUtility_2B96173B", 0x2B96173B, since = 150) { sceUtility_2B96173B(it) }
        registerFunctionInt("sceUtilityGetSystemParamString", 0x34B78343, since = 150) {
            sceUtilityGetSystemParamString(
                int,
                ptr,
                int
            )
        }
        registerFunctionRaw("sceUtility_3AAD51DC", 0x3AAD51DC, since = 150) { sceUtility_3AAD51DC(it) }
        registerFunctionRaw("sceNetplayDialogInitStart", 0x3AD50AE7, since = 150) { sceNetplayDialogInitStart(it) }
        registerFunctionRaw("sceUtilityOskShutdownStart", 0x3DFAEBA9, since = 150) { sceUtilityOskShutdownStart(it) }
        registerFunctionRaw("sceNetplayDialogUpdate", 0x417BED54, since = 150) { sceNetplayDialogUpdate(it) }
        registerFunctionRaw("sceUtilitySetSystemParamString", 0x41E30674, since = 150) {
            sceUtilitySetSystemParamString(
                it
            )
        }
        registerFunctionRaw("sceUtility_42071A83", 0x42071A83, since = 150) { sceUtility_42071A83(it) }
        registerFunctionRaw("sceUtilityGetNetParam", 0x434D4B3A, since = 150) { sceUtilityGetNetParam(it) }
        registerFunctionRaw("sceUtilitySetSystemParamInt", 0x45C18506, since = 150) { sceUtilitySetSystemParamInt(it) }
        registerFunctionRaw("sceUtilityMsgDialogAbort", 0x4928BD96, since = 150) { sceUtilityMsgDialogAbort(it) }
        registerFunctionRaw("sceUtility_4A833BA4", 0x4A833BA4, since = 150) { sceUtility_4A833BA4(it) }
        registerFunctionRaw("sceUtility_4B0A8FE5", 0x4B0A8FE5, since = 150) { sceUtility_4B0A8FE5(it) }
        registerFunctionRaw("sceUtilityOskUpdate", 0x4B85C861, since = 150) { sceUtilityOskUpdate(it) }
        registerFunctionRaw("sceUtilityNetconfInitStart", 0x4DB1E739, since = 150) { sceUtilityNetconfInitStart(it) }
        registerFunctionRaw(
            "sceUtilityGetNetParamLatestID",
            0x4FED24D8,
            since = 150
        ) { sceUtilityGetNetParamLatestID(it) }
        registerFunctionRaw("sceUtility_54A5C62F", 0x54A5C62F, since = 150) { sceUtility_54A5C62F(it) }
        registerFunctionRaw("sceUtilityCheckNetParam", 0x5EEE6548, since = 150) { sceUtilityCheckNetParam(it) }
        registerFunctionRaw("sceUtilityInstallShutdownStart", 0x5EF1C24A, since = 150) {
            sceUtilityInstallShutdownStart(
                it
            )
        }
        registerFunctionRaw("sceUtilityNetconfGetStatus", 0x6332AA39, since = 150) { sceUtilityNetconfGetStatus(it) }
        registerFunctionRaw("sceUtilityUnloadNetModule", 0x64D50C56, since = 150) { sceUtilityUnloadNetModule(it) }
        registerFunctionRaw("sceUtility_6F56F9CF", 0x6F56F9CF, since = 150) { sceUtility_6F56F9CF(it) }
        registerFunctionRaw("sceUtility_70267ADF", 0x70267ADF, since = 150) { sceUtility_70267ADF(it) }
        registerFunctionRaw("sceUtilityGameSharingUpdate", 0x7853182D, since = 150) { sceUtilityGameSharingUpdate(it) }
        registerFunctionRaw("sceUtility_81C44706", 0x81C44706, since = 150) { sceUtility_81C44706(it) }
        registerFunctionRaw("sceUtility_8326AB05", 0x8326AB05, since = 150) { sceUtility_8326AB05(it) }
        registerFunctionRaw("sceUtility_86A03A27", 0x86A03A27, since = 150) { sceUtility_86A03A27(it) }
        registerFunctionRaw("sceUtility_86ABDB1B", 0x86ABDB1B, since = 150) { sceUtility_86ABDB1B(it) }
        registerFunctionRaw("sceUtility_88BC7406", 0x88BC7406, since = 150) { sceUtility_88BC7406(it) }
        registerFunctionRaw("sceUtility_89317C8F", 0x89317C8F, since = 150) { sceUtility_89317C8F(it) }
        registerFunctionRaw("sceUtilityNetconfUpdate", 0x91E70E35, since = 150) { sceUtilityNetconfUpdate(it) }
        registerFunctionRaw("sceUtility_943CBA46", 0x943CBA46, since = 150) { sceUtility_943CBA46(it) }
        registerFunctionRaw("sceUtilityGameSharingGetStatus", 0x946963F3, since = 150) {
            sceUtilityGameSharingGetStatus(
                it
            )
        }
        registerFunctionRaw("sceUtilityMsgDialogUpdate", 0x95FC253B, since = 150) { sceUtilityMsgDialogUpdate(it) }
        registerFunctionRaw(
            "sceUtilitySavedataShutdownStart",
            0x9790B33C,
            since = 150
        ) { sceUtilitySavedataShutdownStart(it) }
        registerFunctionRaw("sceUtilityInstallUpdate", 0xA03D29BA, since = 150) { sceUtilityInstallUpdate(it) }
        registerFunctionRaw("sceUtility_A084E056", 0xA084E056, since = 150) { sceUtility_A084E056(it) }
        registerFunctionRaw("sceUtility_A50E5B30", 0xA50E5B30, since = 150) { sceUtility_A50E5B30(it) }
        registerFunctionRaw("sceUtility_AB083EA9", 0xAB083EA9, since = 150) { sceUtility_AB083EA9(it) }
        registerFunctionRaw("sceUtility_B0FB7FF5", 0xB0FB7FF5, since = 150) { sceUtility_B0FB7FF5(it) }
        registerFunctionRaw("sceUtility_B62A4061", 0xB62A4061, since = 150) { sceUtility_B62A4061(it) }
        registerFunctionRaw("sceNetplayDialogGetStatus", 0xB6CEE597, since = 150) { sceNetplayDialogGetStatus(it) }
        registerFunctionRaw("sceUtility_B8592D5F", 0xB8592D5F, since = 150) { sceUtility_B8592D5F(it) }
        registerFunctionRaw(
            "sceNetplayDialogShutdownStart",
            0xBC6B6296,
            since = 150
        ) { sceNetplayDialogShutdownStart(it) }
        registerFunctionRaw(
            "sceUtilityHtmlViewerGetStatus",
            0xBDA7D894,
            since = 150
        ) { sceUtilityHtmlViewerGetStatus(it) }
        registerFunctionRaw("sceUtilityInstallGetStatus", 0xC4700FA3, since = 150) { sceUtilityInstallGetStatus(it) }
        registerFunctionRaw("sceUtilityGameSharingInitStart", 0xC492F751, since = 150) {
            sceUtilityGameSharingInitStart(
                it
            )
        }
        registerFunctionRaw(
            "sceUtilityHtmlViewerInitStart",
            0xCDC3AA41,
            since = 150
        ) { sceUtilityHtmlViewerInitStart(it) }
        registerFunctionRaw("sceUtility_D17A0573", 0xD17A0573, since = 150) { sceUtility_D17A0573(it) }
        registerFunctionRaw("sceUtilitySavedataUpdate", 0xD4B95FFB, since = 150) { sceUtilitySavedataUpdate(it) }
        registerFunctionRaw("sceUtility_D81957B7", 0xD81957B7, since = 150) { sceUtility_D81957B7(it) }
        registerFunctionRaw("sceUtility_D852CDCE", 0xD852CDCE, since = 150) { sceUtility_D852CDCE(it) }
        registerFunctionRaw("sceUtility_DA97F1AA", 0xDA97F1AA, since = 150) { sceUtility_DA97F1AA(it) }
        registerFunctionRaw("sceUtility_DDE5389D", 0xDDE5389D, since = 150) { sceUtility_DDE5389D(it) }
        registerFunctionRaw("sceUtility_E19C97D6", 0xE19C97D6, since = 150) { sceUtility_E19C97D6(it) }
        registerFunctionRaw("sceUtilityUnloadModule", 0xE49BFE92, since = 150) { sceUtilityUnloadModule(it) }
        registerFunctionRaw("sceUtility_E7B778D8", 0xE7B778D8, since = 150) { sceUtility_E7B778D8(it) }
        registerFunctionRaw("sceUtility_ECE1D3E5", 0xECE1D3E5, since = 150) { sceUtility_ECE1D3E5(it) }
        registerFunctionRaw("sceUtility_ED0FAD38", 0xED0FAD38, since = 150) { sceUtility_ED0FAD38(it) }
        registerFunctionRaw("sceUtility_EF3582B2", 0xEF3582B2, since = 150) { sceUtility_EF3582B2(it) }
        registerFunctionRaw(
            "sceUtilityGameSharingShutdownStart",
            0xEFC6F80F,
            since = 150
        ) { sceUtilityGameSharingShutdownStart(it) }
        registerFunctionRaw("sceUtilityOskGetStatus", 0xF3F76017, since = 150) { sceUtilityOskGetStatus(it) }
        registerFunctionRaw("sceUtility_F3FBC572", 0xF3FBC572, since = 150) { sceUtility_F3FBC572(it) }
        registerFunctionRaw(
            "sceUtilityHtmlViewerShutdownStart",
            0xF5CE1134,
            since = 150
        ) { sceUtilityHtmlViewerShutdownStart(it) }
        registerFunctionRaw("sceUtilityOskInitStart", 0xF6269B82, since = 150) { sceUtilityOskInitStart(it) }
        registerFunctionRaw("sceUtilityUnloadUsbModule", 0xF64910F0, since = 150) { sceUtilityUnloadUsbModule(it) }
        registerFunctionRaw("sceUtilityUnloadAvModule", 0xF7D8D092, since = 150) { sceUtilityUnloadAvModule(it) }
        registerFunctionRaw("sceUtilityNetconfShutdownStart", 0xF88155F6, since = 150) {
            sceUtilityNetconfShutdownStart(
                it
            )
        }
        registerFunctionRaw("sceUtility_F9E0008C", 0xF9E0008C, since = 150) { sceUtility_F9E0008C(it) }
    }
}

enum class PspLanguages(override val id: Int) : IdEnum { // ISO-639-1
    JAPANESE(0), // ja
    ENGLISH(1), // en
    FRENCH(2), // fr
    SPANISH(3), // es
    GERMAN(4), // de
    ITALIAN(5), // it
    DUTCH(6), // nl
    PORTUGUESE(7), // pt
    RUSSIAN(8), // ru
    KOREAN(9), // ko
    TRADITIONAL_CHINESE(10), // zh
    SIMPLIFIED_CHINESE(11);// zh?

    companion object : INT32_ENUM<PspLanguages>(values())
}

enum class PspUtilitySavedataMode(override val id: Int) : IdEnum {
    Autoload(0), // PSP_UTILITY_SAVEDATA_AUTOLOAD = 0
    Autosave(1), // PSP_UTILITY_SAVEDATA_AUTOSAVE = 1
    Load(2), // PSP_UTILITY_SAVEDATA_LOAD = 2
    Save(3), // PSP_UTILITY_SAVEDATA_SAVE = 3
    ListLoad(4), // PSP_UTILITY_SAVEDATA_LISTLOAD = 4
    ListSave(5), // PSP_UTILITY_SAVEDATA_LISTSAVE = 5
    ListDelete(6), // PSP_UTILITY_SAVEDATA_LISTDELETE = 6
    Delete(7), // PSP_UTILITY_SAVEDATA_DELETE = 7
    Sizes(8), // PSP_UTILITY_SAVEDATA_SIZES = 8
    AutoDelete(9), // PSP_UTILITY_SAVEDATA_AUTODELETE = 9
    SingleDelete(10), // PSP_UTILITY_SAVEDATA_SINGLEDELETE = 10 = 0x0A
    List(11), // PSP_UTILITY_SAVEDATA_LIST = 11 = 0x0B
    Files(12), // PSP_UTILITY_SAVEDATA_FILES = 12 = 0x0C
    MakeDataSecure(13), // PSP_UTILITY_SAVEDATA_MAKEDATASECURE = 13 = 0x0D
    MakeData(14), // PSP_UTILITY_SAVEDATA_MAKEDATA = 14 = 0x0E
    ReadSecure(15), // PSP_UTILITY_SAVEDATA_READSECURE = 15 = 0x0F
    Read(16), // PSP_UTILITY_SAVEDATA_READ = 16 = 0x10
    WriteSecure(17), // PSP_UTILITY_SAVEDATA_WRITESECURE = 17 = 0x11
    Write(18), // PSP_UTILITY_SAVEDATA_WRITE = 18 = 0x12
    EraseSecure(19), // PSP_UTILITY_SAVEDATA_ERASESECURE = 19 = 0x13
    Erase(20), // PSP_UTILITY_SAVEDATA_ERASE = 20 = 0x14
    DeleteData(21), // PSP_UTILITY_SAVEDATA_DELETEDATA = 21 = 0x15
    GetSize(22); // PSP_UTILITY_SAVEDATA_GETSIZE = 22 = 0x16

    companion object : INT32_ENUM<PspUtilitySavedataMode>(values())
}

enum class PspUtilitySavedataFocus(override val id: Int) : IdEnum {
    PSP_UTILITY_SAVEDATA_FOCUS_UNKNOWN(0), //
    PSP_UTILITY_SAVEDATA_FOCUS_FIRSTLIST(1), // First in list
    PSP_UTILITY_SAVEDATA_FOCUS_LASTLIST(2), // Last in list
    PSP_UTILITY_SAVEDATA_FOCUS_LATEST(3), // Most recent date
    PSP_UTILITY_SAVEDATA_FOCUS_OLDEST(4), // Oldest date
    PSP_UTILITY_SAVEDATA_FOCUS_UNKNOWN2(5), //
    PSP_UTILITY_SAVEDATA_FOCUS_UNKNOWN3(6), //
    PSP_UTILITY_SAVEDATA_FOCUS_FIRSTEMPTY(7), // First empty slot
    PSP_UTILITY_SAVEDATA_FOCUS_LASTEMPTY(8); // Last empty slot

    companion object : INT32_ENUM<PspUtilitySavedataFocus>(values())
}

data class PspUtilityDialogCommon(
    var size: Int = 0, // 0000 - Size of the structure
    var language: PspLanguages = PspLanguages.SPANISH, // 0004 - Language
    var buttonSwap: Int = 0, // 0008 - Set to 1 for X/O button swap
    var graphicsThread: Int = 0, // 000C - Graphics thread priority
    var accessThread: Int = 0, // 0010 - Access/fileio thread priority (SceJobThread)
    var fontThread: Int = 0, // 0014 - Font thread priority (ScePafThread)
    var soundThread: Int = 0, // 0018 - Sound thread priority
    var result: Int = SceKernelErrors.ERROR_OK, // 001C - Result
    var reserved: ArrayList<Int> = arrayListOf(0, 0, 0, 0) // 0020 - Set to 0
) {
    companion object : Struct<PspUtilityDialogCommon>(
        { PspUtilityDialogCommon() },
        PspUtilityDialogCommon::size AS INT32,
        PspUtilityDialogCommon::language AS PspLanguages,
        PspUtilityDialogCommon::buttonSwap AS INT32,
        PspUtilityDialogCommon::graphicsThread AS INT32,
        PspUtilityDialogCommon::accessThread AS INT32,
        PspUtilityDialogCommon::fontThread AS INT32,
        PspUtilityDialogCommon::soundThread AS INT32,
        PspUtilityDialogCommon::result AS INT32,
        PspUtilityDialogCommon::reserved AS ARRAY(INT32, 4)
    )
}

data class PspUtilitySavedataSFOParam(
    var title: String = "", // 0000 -
    var savedataTitle: String = "", // 0080 -
    var detail: String = "", // 0100 -
    var parentalLevel: Int = 0, // 0500 -
    var unknown: ArrayList<Int> = arrayListOf(0, 0, 0) // 0501 -
) {
    companion object : Struct<PspUtilitySavedataSFOParam>(
        { PspUtilitySavedataSFOParam() },
        PspUtilitySavedataSFOParam::title AS STRINGZ(0x80),
        PspUtilitySavedataSFOParam::savedataTitle AS STRINGZ(0x80),
        PspUtilitySavedataSFOParam::detail AS STRINGZ(0x400),
        PspUtilitySavedataSFOParam::parentalLevel AS UINT8,
        PspUtilitySavedataSFOParam::unknown AS ARRAY(UINT8, 3)
    )
}

data class PspUtilitySavedataFileData(
    var bufferPointer: Int = 0, // 0000 -
    var bufferSize: Int = 0, // 0004 -
    var size: Int = 0, // 0008 - why are there two sizes?
    var unknown: Int = 0 // 000C -
) {
    val used: Boolean
        get() {
            if (this.bufferPointer == 0) return false
            //if (BufferSize == 0) return false;
            if (this.size == 0) return false
            return true
        }

    companion object : Struct<PspUtilitySavedataFileData>(
        { PspUtilitySavedataFileData() },
        PspUtilitySavedataFileData::bufferPointer AS INT32,
        PspUtilitySavedataFileData::bufferSize AS INT32,
        PspUtilitySavedataFileData::size AS INT32,
        PspUtilitySavedataFileData::unknown AS INT32
    )
}

data class SceUtilitySavedataParam(
    var base: PspUtilityDialogCommon = PspUtilityDialogCommon(), // 0000 - PspUtilityDialogCommon
    var mode: PspUtilitySavedataMode = PspUtilitySavedataMode.Autoload, // 0030 -
    var unknown1: Int = 0, // 0034 -
    var overwrite: Int = 0, // 0038 -
    var gameName: String = "", // 003C - GameName: name used from the game for saves, equal for all saves
    var saveName: String = "", // 004C - SaveName: name of the particular save, normally a number
    var saveNameListPointer: Int = 0, // 0060 - SaveNameList: used by multiple modes (char[20])
    var fileName: String = "", // 0064 - FileName: Name of the data file of the game for example DATA.BIN
    var dataBufPointer: Int = 0, // 0074 - Pointer to a buffer that will contain data file unencrypted data
    var dataBufSize: Int = 0, // 0078 - Size of allocated space to dataBuf
    var dataSize: Int = 0, // 007C -
    var sfoParam: PspUtilitySavedataSFOParam = PspUtilitySavedataSFOParam(), // 0080 - (504?)
    var icon0FileData: PspUtilitySavedataFileData = PspUtilitySavedataFileData(), // 0584 - (16)
    var icon1FileData: PspUtilitySavedataFileData = PspUtilitySavedataFileData(), // 0594 - (16)
    var pic1FileData: PspUtilitySavedataFileData = PspUtilitySavedataFileData(), // 05A4 - (16)
    var snd0FileData: PspUtilitySavedataFileData = PspUtilitySavedataFileData(), // 05B4 - (16)
    var newDataPointer: Int = 0, // 05C4 -Pointer to an PspUtilitySavedataListSaveNewData structure (PspUtilitySavedataListSaveNewData *)
    var focus: PspUtilitySavedataFocus = PspUtilitySavedataFocus.PSP_UTILITY_SAVEDATA_FOCUS_UNKNOWN, // 05C8 -Initial focus for lists
    var abortStatus: Int = 0, // 05CC -
    var msFreeAddr: Int = 0, // 05D0 -
    var msDataAddr: Int = 0, // 05D4 -
    var utilityDataAddr: Int = 0, // 05D8 -
    var key: ArrayList<Int> = arrayListOf(
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0
    ), // 05E0 - Key: Encrypt/decrypt key for save with firmware >= 2.00
    var secureVersion: Int = 0, // 05F0 -
    var multiStatus: Int = 0, // 05F4 -
    var idListAddr: Int = 0, // 05F8 -
    var fileListAddr: Int = 0, // 05FC -
    var sizeAddr: Int = 0, // 0600 -
    var unknown3: ArrayList<Int> = arrayListOf(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0) // 0604 -unknown3: ?
) {
    companion object : Struct<SceUtilitySavedataParam>(
        { SceUtilitySavedataParam() },
        SceUtilitySavedataParam::base AS PspUtilityDialogCommon,
        SceUtilitySavedataParam::mode AS PspUtilitySavedataMode,
        SceUtilitySavedataParam::unknown1 AS INT32,
        SceUtilitySavedataParam::overwrite AS INT32,
        SceUtilitySavedataParam::gameName AS STRINGZ(16),
        SceUtilitySavedataParam::saveName AS STRINGZ(20),
        SceUtilitySavedataParam::saveNameListPointer AS INT32,
        SceUtilitySavedataParam::fileName AS STRINGZ(16),
        SceUtilitySavedataParam::dataBufPointer AS INT32,
        SceUtilitySavedataParam::dataBufSize AS INT32,
        SceUtilitySavedataParam::dataSize AS INT32,
        SceUtilitySavedataParam::sfoParam AS PspUtilitySavedataSFOParam,
        SceUtilitySavedataParam::icon0FileData AS PspUtilitySavedataFileData,
        SceUtilitySavedataParam::icon1FileData AS PspUtilitySavedataFileData,
        SceUtilitySavedataParam::pic1FileData AS PspUtilitySavedataFileData,
        SceUtilitySavedataParam::snd0FileData AS PspUtilitySavedataFileData,
        SceUtilitySavedataParam::newDataPointer AS INT32,
        SceUtilitySavedataParam::focus AS PspUtilitySavedataFocus,
        SceUtilitySavedataParam::abortStatus AS INT32,
        SceUtilitySavedataParam::msFreeAddr AS INT32,
        SceUtilitySavedataParam::msDataAddr AS INT32,
        SceUtilitySavedataParam::utilityDataAddr AS INT32,
        SceUtilitySavedataParam::key AS ARRAY(UINT8, 16),
        SceUtilitySavedataParam::secureVersion AS INT32,
        SceUtilitySavedataParam::multiStatus AS INT32,
        SceUtilitySavedataParam::idListAddr AS INT32,
        SceUtilitySavedataParam::fileListAddr AS INT32,
        SceUtilitySavedataParam::sizeAddr AS INT32,
        SceUtilitySavedataParam::unknown3 AS ARRAY(UINT8, 20 - 5)
    )
}

enum class PspUtilityMsgDialogMode(override val id: Int) : IdEnum {
    PSP_UTILITY_MSGDIALOG_MODE_ERROR(0), // Error message
    PSP_UTILITY_MSGDIALOG_MODE_TEXT(1); // String message

    companion object : INT32_ENUM<PspUtilityMsgDialogMode>(values())
}

enum class PspUtilityMsgDialogOption(override val id: Int) : IdEnum {
    PSP_UTILITY_MSGDIALOG_OPTION_ERROR(0x00000000), // Error message (why two flags?)
    PSP_UTILITY_MSGDIALOG_OPTION_TEXT(0x00000001), // Text message (why two flags?)
    PSP_UTILITY_MSGDIALOG_OPTION_YESNO_BUTTONS(0x00000010), // Yes/No buttons instead of 'Cancel'
    PSP_UTILITY_MSGDIALOG_OPTION_DEFAULT_NO(0x00000100); // Default position 'No', if not set will default to 'Yes'

    companion object : INT32_ENUM<PspUtilityMsgDialogOption>(values())
}

enum class PspUtilityMsgDialogPressed(override val id: Int) : IdEnum {
    PSP_UTILITY_MSGDIALOG_RESULT_UNKNOWN1(0),
    PSP_UTILITY_MSGDIALOG_RESULT_YES(1),
    PSP_UTILITY_MSGDIALOG_RESULT_NO(2),
    PSP_UTILITY_MSGDIALOG_RESULT_BACK(3);

    companion object : INT32_ENUM<PspUtilityMsgDialogPressed>(values())
}

enum class DialogStepEnum(override val id: Int) : IdEnum {
    NONE(0),
    INIT(1),
    PROCESSING(2),
    SUCCESS(3),
    SHUTDOWN(4);

    companion object : INT32_ENUM<DialogStepEnum>(values())
}

data class PspUtilityMsgDialogParams(
    var base: PspUtilityDialogCommon = PspUtilityDialogCommon(),
    var unknown: Int = 0, // uint
    var mnode: PspUtilityMsgDialogMode = PspUtilityMsgDialogMode.PSP_UTILITY_MSGDIALOG_MODE_ERROR, // uint
    var errorValue: Int = 0, // uint
    var message: String = "", // byte[512]
    var options: PspUtilityMsgDialogOption = PspUtilityMsgDialogOption.PSP_UTILITY_MSGDIALOG_OPTION_DEFAULT_NO,
    var buttonPressed: PspUtilityMsgDialogPressed = PspUtilityMsgDialogPressed.PSP_UTILITY_MSGDIALOG_RESULT_BACK
) {
    companion object : Struct<PspUtilityMsgDialogParams>(
        { PspUtilityMsgDialogParams() },
        PspUtilityMsgDialogParams::base AS PspUtilityDialogCommon,
        PspUtilityMsgDialogParams::unknown AS INT32,
        PspUtilityMsgDialogParams::mnode AS PspUtilityMsgDialogMode,
        PspUtilityMsgDialogParams::errorValue AS INT32,
        PspUtilityMsgDialogParams::message AS STRINGZ(512),
        PspUtilityMsgDialogParams::options AS PspUtilityMsgDialogOption,
        PspUtilityMsgDialogParams::buttonPressed AS PspUtilityMsgDialogPressed
    )
}