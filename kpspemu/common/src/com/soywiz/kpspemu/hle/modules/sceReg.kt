package com.soywiz.kpspemu.hle.modules

import com.soywiz.kpspemu.*
import com.soywiz.kpspemu.cpu.*
import com.soywiz.kpspemu.hle.*
import com.soywiz.kpspemu.mem.*
import com.soywiz.kpspemu.util.*

class sceReg(emulator: Emulator) : SceModule(emulator, "sceReg", 0x40010011, "registry.prx", "sceRegistry_Service") {
    fun sceRegOpenRegistry(regParamPtr: Ptr, mode: Int, regHandlePtr: Ptr): Int {
        val param = regParamPtr.read(RegParam)
        logger.error { "sceRegOpenRegistry:$param,$mode,$regHandlePtr" }
        regHandlePtr.sw(0, 0)
        return 0
    }

    fun sceRegOpenCategory(regHandle: Int, name: String?, mode: Int, regCategoryHandlePtr: Ptr): Int {
        logger.error { "sceRegOpenCategory:$regHandle,$name,$mode,$regCategoryHandlePtr" }
        return 0
    }

    fun sceRegGetKeyInfo(
        categoryHandle: Int,
        name: String?,
        regKeyHandlePtr: Ptr,
        regKeyTypesPtr: Ptr,
        sizePtr: Ptr
    ): Int {
        logger.error { "sceRegGetKeyInfo:$categoryHandle,$name,$regKeyHandlePtr,$regKeyTypesPtr,$sizePtr" }
        return 0
    }

    fun sceRegGetKeyValue(categoryHandle: Int, regKeyHandle: Int, bufferPtr: Ptr, size: Int): Int {
        logger.error { "sceRegGetKeyValue:$categoryHandle,$regKeyHandle,$bufferPtr,$size" }
        return 0
    }

    fun sceRegFlushCategory(categoryHandle: Int): Int {
        logger.error { "sceRegFlushCategory:$categoryHandle" }
        return 0
    }

    fun sceRegCloseCategory(categoryHandle: Int): Int {
        logger.error { "sceRegCloseCategory:$categoryHandle" }
        return 0
    }

    fun sceRegFlushRegistry(regHandle: Int): Int {
        logger.error { "sceRegFlushRegistry:$regHandle" }
        return 0
    }

    fun sceRegCloseRegistry(regHandle: Int): Int {
        logger.error { "sceRegCloseRegistry:$regHandle" }
        return 0
    }

    fun sceRegSetKeyValue(cpu: CpuState): Unit = UNIMPLEMENTED(0x17768E14)
    fun sceRegGetKeysNum(cpu: CpuState): Unit = UNIMPLEMENTED(0x2C0DB9DD)
    fun sceRegGetKeys(cpu: CpuState): Unit = UNIMPLEMENTED(0x2D211135)
    fun sceRegGetKeyValueByName(cpu: CpuState): Unit = UNIMPLEMENTED(0x30BE0259)
    fun sceRegRemoveKey(cpu: CpuState): Unit = UNIMPLEMENTED(0x3615BC87)
    fun sceRegRemoveCategory(cpu: CpuState): Unit = UNIMPLEMENTED(0x4CA16893)
    fun sceRegCreateKey(cpu: CpuState): Unit = UNIMPLEMENTED(0x57641A81)
    fun sceReg_835ECE6F(cpu: CpuState): Unit = UNIMPLEMENTED(0x835ECE6F)
    fun sceRegExit(cpu: CpuState): Unit = UNIMPLEMENTED(0x9B25EDF1)
    fun sceReg_BE8C1263(cpu: CpuState): Unit = UNIMPLEMENTED(0xBE8C1263)
    fun sceRegGetKeyInfoByName(cpu: CpuState): Unit = UNIMPLEMENTED(0xC5768D02)
    fun sceRegRemoveRegistry(cpu: CpuState): Unit = UNIMPLEMENTED(0xDEDA92BF)


    override fun registerModule() {
        registerFunctionInt("sceRegOpenRegistry", 0x92E41280, since = 150) { sceRegOpenRegistry(ptr, int, ptr) }
        registerFunctionInt("sceRegOpenCategory", 0x1D8A762E, since = 150) { sceRegOpenCategory(int, str, int, ptr) }
        registerFunctionInt("sceRegGetKeyInfo", 0xD4475AA8, since = 150) { sceRegGetKeyInfo(int, str, ptr, ptr, ptr) }
        registerFunctionInt("sceRegGetKeyValue", 0x28A8E98A, since = 150) { sceRegGetKeyValue(int, int, ptr, int) }
        registerFunctionInt("sceRegFlushCategory", 0x0D69BF40, since = 150) { sceRegFlushCategory(int) }
        registerFunctionInt("sceRegCloseCategory", 0x0CAE832B, since = 150) { sceRegCloseCategory(int) }
        registerFunctionInt("sceRegFlushRegistry", 0x39461B4D, since = 150) { sceRegFlushRegistry(int) }
        registerFunctionInt("sceRegCloseRegistry", 0xFA8A5739, since = 150) { sceRegCloseRegistry(int) }

        registerFunctionRaw("sceRegSetKeyValue", 0x17768E14, since = 150) { sceRegSetKeyValue(it) }
        registerFunctionRaw("sceRegGetKeysNum", 0x2C0DB9DD, since = 150) { sceRegGetKeysNum(it) }
        registerFunctionRaw("sceRegGetKeys", 0x2D211135, since = 150) { sceRegGetKeys(it) }
        registerFunctionRaw("sceRegGetKeyValueByName", 0x30BE0259, since = 150) { sceRegGetKeyValueByName(it) }
        registerFunctionRaw("sceRegRemoveKey", 0x3615BC87, since = 150) { sceRegRemoveKey(it) }
        registerFunctionRaw("sceRegRemoveCategory", 0x4CA16893, since = 150) { sceRegRemoveCategory(it) }
        registerFunctionRaw("sceRegCreateKey", 0x57641A81, since = 150) { sceRegCreateKey(it) }
        registerFunctionRaw("sceReg_835ECE6F", 0x835ECE6F, since = 150) { sceReg_835ECE6F(it) }
        registerFunctionRaw("sceRegExit", 0x9B25EDF1, since = 150) { sceRegExit(it) }
        registerFunctionRaw("sceReg_BE8C1263", 0xBE8C1263, since = 150) { sceReg_BE8C1263(it) }
        registerFunctionRaw("sceRegGetKeyInfoByName", 0xC5768D02, since = 150) { sceRegGetKeyInfoByName(it) }
        registerFunctionRaw("sceRegRemoveRegistry", 0xDEDA92BF, since = 150) { sceRegRemoveRegistry(it) }
    }
}

data class RegParam(
    var regType: Int = 0,
    var name: String = "",
    var nameLength: Int = 0,
    var unknown2: Int = 0,
    var unknown3: Int = 0
) {
    companion object : Struct<RegParam>(
        { RegParam() },
        RegParam::regType AS INT32,
        RegParam::name AS STRINGZ(256),
        RegParam::nameLength AS INT32,
        RegParam::unknown2 AS INT32,
        RegParam::unknown3 AS INT32
    )
}
