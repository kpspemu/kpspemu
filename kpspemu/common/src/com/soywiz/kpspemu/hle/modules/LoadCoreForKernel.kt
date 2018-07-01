package com.soywiz.kpspemu.hle.modules

import com.soywiz.kpspemu.*
import com.soywiz.kpspemu.cpu.*
import com.soywiz.kpspemu.hle.*

@Suppress("UNUSED_PARAMETER")
class LoadCoreForKernel(emulator: Emulator) :
    SceModule(emulator, "LoadCoreForKernel", 0x00010011, "loadcore.prx", "sceLoaderCore") {
    fun sceKernelIcacheClearAll(): Unit = emulator.invalidateInstructionCache()

    fun LoadCoreForKernel_00E94E85(cpu: CpuState): Unit = UNIMPLEMENTED(0x00E94E85)
    fun LoadCoreForKernel_0B53340F(cpu: CpuState): Unit = UNIMPLEMENTED(0x0B53340F)
    fun LoadCoreForKernel_0DDEC402(cpu: CpuState): Unit = UNIMPLEMENTED(0x0DDEC402)
    fun LoadCoreForKernel_0F9DDF1D(cpu: CpuState): Unit = UNIMPLEMENTED(0x0F9DDF1D)
    fun LoadCoreForKernel_10FD7D37(cpu: CpuState): Unit = UNIMPLEMENTED(0x10FD7D37)
    fun LoadCoreForKernel_115FA474(cpu: CpuState): Unit = UNIMPLEMENTED(0x115FA474)
    fun LoadCoreForKernel_1285603B(cpu: CpuState): Unit = UNIMPLEMENTED(0x1285603B)
    fun LoadCoreForKernel_14C57306(cpu: CpuState): Unit = UNIMPLEMENTED(0x14C57306)
    fun LoadCoreForKernel_23C81B66(cpu: CpuState): Unit = UNIMPLEMENTED(0x23C81B66)
    fun LoadCoreForKernel_2FAD3D30(cpu: CpuState): Unit = UNIMPLEMENTED(0x2FAD3D30)
    fun LoadCoreForKernel_358CEBF7(cpu: CpuState): Unit = UNIMPLEMENTED(0x358CEBF7)
    fun LoadCoreForKernel_386369BB(cpu: CpuState): Unit = UNIMPLEMENTED(0x386369BB)
    fun LoadCoreForKernel_3E9E20D7(cpu: CpuState): Unit = UNIMPLEMENTED(0x3E9E20D7)
    fun LoadCoreForKernel_3F567499(cpu: CpuState): Unit = UNIMPLEMENTED(0x3F567499)
    fun LoadCoreForKernel_587F4973(cpu: CpuState): Unit = UNIMPLEMENTED(0x587F4973)
    fun LoadCoreForKernel_618C92FF(cpu: CpuState): Unit = UNIMPLEMENTED(0x618C92FF)
    fun LoadCoreForKernel_65BE4168(cpu: CpuState): Unit = UNIMPLEMENTED(0x65BE4168)
    fun LoadCoreForKernel_6B3E192B(cpu: CpuState): Unit = UNIMPLEMENTED(0x6B3E192B)
    fun LoadCoreForKernel_6C00BE57(cpu: CpuState): Unit = UNIMPLEMENTED(0x6C00BE57)
    fun LoadCoreForKernel_77E3CB6B(cpu: CpuState): Unit = UNIMPLEMENTED(0x77E3CB6B)
    fun LoadCoreForKernel_9EEC51A4(cpu: CpuState): Unit = UNIMPLEMENTED(0x9EEC51A4)
    fun LoadCoreForKernel_9F5A77CB(cpu: CpuState): Unit = UNIMPLEMENTED(0x9F5A77CB)
    fun LoadCoreForKernel_A48D2D08(cpu: CpuState): Unit = UNIMPLEMENTED(0xA48D2D08)
    fun LoadCoreForKernel_AE96A41B(cpu: CpuState): Unit = UNIMPLEMENTED(0xAE96A41B)
    fun LoadCoreForKernel_B3391485(cpu: CpuState): Unit = UNIMPLEMENTED(0xB3391485)
    fun LoadCoreForKernel_B5D16A21(cpu: CpuState): Unit = UNIMPLEMENTED(0xB5D16A21)
    fun LoadCoreForKernel_B63183B5(cpu: CpuState): Unit = UNIMPLEMENTED(0xB63183B5)
    fun LoadCoreForKernel_BFAD9D71(cpu: CpuState): Unit = UNIMPLEMENTED(0xBFAD9D71)
    fun LoadCoreForKernel_C237D677(cpu: CpuState): Unit = UNIMPLEMENTED(0xC237D677)
    fun LoadCoreForKernel_CA2BC850(cpu: CpuState): Unit = UNIMPLEMENTED(0xCA2BC850)
    fun LoadCoreForKernel_CE2B6937(cpu: CpuState): Unit = UNIMPLEMENTED(0xCE2B6937)
    fun LoadCoreForKernel_D44501A6(cpu: CpuState): Unit = UNIMPLEMENTED(0xD44501A6)
    fun LoadCoreForKernel_DB6BAA71(cpu: CpuState): Unit = UNIMPLEMENTED(0xDB6BAA71)
    fun LoadCoreForKernel_DD303D79(cpu: CpuState): Unit = UNIMPLEMENTED(0xDD303D79)
    fun LoadCoreForKernel_E056884A(cpu: CpuState): Unit = UNIMPLEMENTED(0xE056884A)
    fun LoadCoreForKernel_EA9500BC(cpu: CpuState): Unit = UNIMPLEMENTED(0xEA9500BC)
    fun LoadCoreForKernel_EC2861D0(cpu: CpuState): Unit = UNIMPLEMENTED(0xEC2861D0)
    fun LoadCoreForKernel_F1130D5F(cpu: CpuState): Unit = UNIMPLEMENTED(0xF1130D5F)
    fun LoadCoreForKernel_F1ACA4B2(cpu: CpuState): Unit = UNIMPLEMENTED(0xF1ACA4B2)
    fun LoadCoreForKernel_F6C7F05C(cpu: CpuState): Unit = UNIMPLEMENTED(0xF6C7F05C)


    override fun registerModule() {
        registerFunctionVoid("sceKernelIcacheClearAll", 0xD8779AC6, since = 150) { sceKernelIcacheClearAll() }

        registerFunctionRaw("LoadCoreForKernel_00E94E85", 0x00E94E85, since = 150) { LoadCoreForKernel_00E94E85(it) }
        registerFunctionRaw("LoadCoreForKernel_0B53340F", 0x0B53340F, since = 150) { LoadCoreForKernel_0B53340F(it) }
        registerFunctionRaw("LoadCoreForKernel_0DDEC402", 0x0DDEC402, since = 150) { LoadCoreForKernel_0DDEC402(it) }
        registerFunctionRaw("LoadCoreForKernel_0F9DDF1D", 0x0F9DDF1D, since = 150) { LoadCoreForKernel_0F9DDF1D(it) }
        registerFunctionRaw("LoadCoreForKernel_10FD7D37", 0x10FD7D37, since = 150) { LoadCoreForKernel_10FD7D37(it) }
        registerFunctionRaw("LoadCoreForKernel_115FA474", 0x115FA474, since = 150) { LoadCoreForKernel_115FA474(it) }
        registerFunctionRaw("LoadCoreForKernel_1285603B", 0x1285603B, since = 150) { LoadCoreForKernel_1285603B(it) }
        registerFunctionRaw("LoadCoreForKernel_14C57306", 0x14C57306, since = 150) { LoadCoreForKernel_14C57306(it) }
        registerFunctionRaw("LoadCoreForKernel_23C81B66", 0x23C81B66, since = 150) { LoadCoreForKernel_23C81B66(it) }
        registerFunctionRaw("LoadCoreForKernel_2FAD3D30", 0x2FAD3D30, since = 150) { LoadCoreForKernel_2FAD3D30(it) }
        registerFunctionRaw("LoadCoreForKernel_358CEBF7", 0x358CEBF7, since = 150) { LoadCoreForKernel_358CEBF7(it) }
        registerFunctionRaw("LoadCoreForKernel_386369BB", 0x386369BB, since = 150) { LoadCoreForKernel_386369BB(it) }
        registerFunctionRaw("LoadCoreForKernel_3E9E20D7", 0x3E9E20D7, since = 150) { LoadCoreForKernel_3E9E20D7(it) }
        registerFunctionRaw("LoadCoreForKernel_3F567499", 0x3F567499, since = 150) { LoadCoreForKernel_3F567499(it) }
        registerFunctionRaw("LoadCoreForKernel_587F4973", 0x587F4973, since = 150) { LoadCoreForKernel_587F4973(it) }
        registerFunctionRaw("LoadCoreForKernel_618C92FF", 0x618C92FF, since = 150) { LoadCoreForKernel_618C92FF(it) }
        registerFunctionRaw("LoadCoreForKernel_65BE4168", 0x65BE4168, since = 150) { LoadCoreForKernel_65BE4168(it) }
        registerFunctionRaw("LoadCoreForKernel_6B3E192B", 0x6B3E192B, since = 150) { LoadCoreForKernel_6B3E192B(it) }
        registerFunctionRaw("LoadCoreForKernel_6C00BE57", 0x6C00BE57, since = 150) { LoadCoreForKernel_6C00BE57(it) }
        registerFunctionRaw("LoadCoreForKernel_77E3CB6B", 0x77E3CB6B, since = 150) { LoadCoreForKernel_77E3CB6B(it) }
        registerFunctionRaw("LoadCoreForKernel_9EEC51A4", 0x9EEC51A4, since = 150) { LoadCoreForKernel_9EEC51A4(it) }
        registerFunctionRaw("LoadCoreForKernel_9F5A77CB", 0x9F5A77CB, since = 150) { LoadCoreForKernel_9F5A77CB(it) }
        registerFunctionRaw("LoadCoreForKernel_A48D2D08", 0xA48D2D08, since = 150) { LoadCoreForKernel_A48D2D08(it) }
        registerFunctionRaw("LoadCoreForKernel_AE96A41B", 0xAE96A41B, since = 150) { LoadCoreForKernel_AE96A41B(it) }
        registerFunctionRaw("LoadCoreForKernel_B3391485", 0xB3391485, since = 150) { LoadCoreForKernel_B3391485(it) }
        registerFunctionRaw("LoadCoreForKernel_B5D16A21", 0xB5D16A21, since = 150) { LoadCoreForKernel_B5D16A21(it) }
        registerFunctionRaw("LoadCoreForKernel_B63183B5", 0xB63183B5, since = 150) { LoadCoreForKernel_B63183B5(it) }
        registerFunctionRaw("LoadCoreForKernel_BFAD9D71", 0xBFAD9D71, since = 150) { LoadCoreForKernel_BFAD9D71(it) }
        registerFunctionRaw("LoadCoreForKernel_C237D677", 0xC237D677, since = 150) { LoadCoreForKernel_C237D677(it) }
        registerFunctionRaw("LoadCoreForKernel_CA2BC850", 0xCA2BC850, since = 150) { LoadCoreForKernel_CA2BC850(it) }
        registerFunctionRaw("LoadCoreForKernel_CE2B6937", 0xCE2B6937, since = 150) { LoadCoreForKernel_CE2B6937(it) }
        registerFunctionRaw("LoadCoreForKernel_D44501A6", 0xD44501A6, since = 150) { LoadCoreForKernel_D44501A6(it) }
        registerFunctionRaw("LoadCoreForKernel_DB6BAA71", 0xDB6BAA71, since = 150) { LoadCoreForKernel_DB6BAA71(it) }
        registerFunctionRaw("LoadCoreForKernel_DD303D79", 0xDD303D79, since = 150) { LoadCoreForKernel_DD303D79(it) }
        registerFunctionRaw("LoadCoreForKernel_E056884A", 0xE056884A, since = 150) { LoadCoreForKernel_E056884A(it) }
        registerFunctionRaw("LoadCoreForKernel_EA9500BC", 0xEA9500BC, since = 150) { LoadCoreForKernel_EA9500BC(it) }
        registerFunctionRaw("LoadCoreForKernel_EC2861D0", 0xEC2861D0, since = 150) { LoadCoreForKernel_EC2861D0(it) }
        registerFunctionRaw("LoadCoreForKernel_F1130D5F", 0xF1130D5F, since = 150) { LoadCoreForKernel_F1130D5F(it) }
        registerFunctionRaw("LoadCoreForKernel_F1ACA4B2", 0xF1ACA4B2, since = 150) { LoadCoreForKernel_F1ACA4B2(it) }
        registerFunctionRaw("LoadCoreForKernel_F6C7F05C", 0xF6C7F05C, since = 150) { LoadCoreForKernel_F6C7F05C(it) }
    }
}
