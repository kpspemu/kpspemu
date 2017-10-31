package com.soywiz.kpspemu.hle.modules

import com.soywiz.korio.error.invalidOp
import com.soywiz.kpspemu.Emulator
import com.soywiz.kpspemu.cpu.CpuState
import com.soywiz.kpspemu.hle.SceModule
import com.soywiz.kpspemu.hle.manager.MemoryAnchor
import com.soywiz.kpspemu.hle.manager.MemoryPartition
import com.soywiz.kpspemu.hle.manager.OutOfMemoryError
import com.soywiz.kpspemu.memoryManager
import com.soywiz.kpspemu.util.ResourceItem
import com.soywiz.kpspemu.util.ResourceList
import com.soywiz.kpspemu.util.hex

@Suppress("UNUSED_PARAMETER")
class SysMemUserForUser(emulator: Emulator) : SceModule(emulator, "SysMemUserForUser", 0x40000011, "sysmem.prx", "sceSystemMemoryManager") {
	class Partition(override val id: Int) : ResourceItem {
		lateinit var part: MemoryPartition
	}

	val partitions = ResourceList("Partition") { Partition(it) }

	fun sceKernelAllocPartitionMemory(partitionId: Int, name: String?, anchor: Int, size: Int, address: Int): Int {
		println("WIP: sceKernelAllocPartitionMemory($partitionId, $name, $anchor, $size, ${address.hex})")
		try {
			val parentPartition = memoryManager.memoryPartitionsUid[partitionId] ?: invalidOp("Invalid partition $partitionId")
			val allocatedPartition = parentPartition.allocate(size.toDouble(), MemoryAnchor(anchor), address.toDouble(), name ?: "block")
			return partitions.alloc().apply {
				part = allocatedPartition
			}.id
		} catch (e: OutOfMemoryError) {
			//console.error(e)
			//return SceKernelErrors.ERROR_KERNEL_FAILED_ALLOC_MEMBLOCK
			return -1
		}
	}

	fun sceKernelGetBlockHeadAddr(partitionId: Int): Int {
		val partition = partitions[partitionId]
		return partition.part.low.toInt()
	}

	fun SysMemUserForUser_057E7380(cpu: CpuState): Unit = UNIMPLEMENTED(0x057E7380)
	fun sceKernelPrintf(cpu: CpuState): Unit = UNIMPLEMENTED(0x13A5ABEF)
	fun sceKernelQueryMemoryInfo(cpu: CpuState): Unit = UNIMPLEMENTED(0x2A3E5280)
	fun SysMemUserForUser_315AD3A0(cpu: CpuState): Unit = UNIMPLEMENTED(0x315AD3A0)
	fun SysMemUserForUser_342061E5(cpu: CpuState): Unit = UNIMPLEMENTED(0x342061E5)
	fun sceKernelDevkitVersion(cpu: CpuState): Unit = UNIMPLEMENTED(0x3FC9AE6A)
	fun SysMemUserForUser_50F61D8A(cpu: CpuState): Unit = UNIMPLEMENTED(0x50F61D8A)
	fun sceKernelSetCompiledSdkVersion(cpu: CpuState): Unit = UNIMPLEMENTED(0x7591C7DB)
	fun SysMemUserForUser_91DE343C(cpu: CpuState): Unit = UNIMPLEMENTED(0x91DE343C)
	fun sceKernelMaxFreeMemSize(cpu: CpuState): Unit = UNIMPLEMENTED(0xA291F107)
	fun SysMemUserForUser_A6848DF8(cpu: CpuState): Unit = UNIMPLEMENTED(0xA6848DF8)
	fun SysMemUserForUser_ACBD88CA(cpu: CpuState): Unit = UNIMPLEMENTED(0xACBD88CA)
	fun sceKernelFreePartitionMemory(cpu: CpuState): Unit = UNIMPLEMENTED(0xB6D61D02)
	fun SysMemUserForUser_D8DE5C1E(cpu: CpuState): Unit = UNIMPLEMENTED(0xD8DE5C1E)
	fun SysMemUserForUser_DB83A952(cpu: CpuState): Unit = UNIMPLEMENTED(0xDB83A952)
	fun SysMemUserForUser_EBD5C3E6(cpu: CpuState): Unit = UNIMPLEMENTED(0xEBD5C3E6)
	fun sceKernelSetCompilerVersion(cpu: CpuState): Unit = UNIMPLEMENTED(0xF77D77CB)
	fun sceKernelTotalFreeMemSize(cpu: CpuState): Unit = UNIMPLEMENTED(0xF919F628)
	fun sceKernelGetCompiledSdkVersion(cpu: CpuState): Unit = UNIMPLEMENTED(0xFC114573)
	fun SysMemUserForUser_FE707FDF(cpu: CpuState): Unit = UNIMPLEMENTED(0xFE707FDF)


	override fun registerModule() {
		registerFunctionInt("sceKernelAllocPartitionMemory", 0x237DBD4F, since = 150) { sceKernelAllocPartitionMemory(int, string, int, int, int) }
		registerFunctionInt("sceKernelGetBlockHeadAddr", 0x9D9A5BA1, since = 150) { sceKernelGetBlockHeadAddr(int) }

		registerFunctionRaw("SysMemUserForUser_057E7380", 0x057E7380, since = 150) { SysMemUserForUser_057E7380(it) }
		registerFunctionRaw("sceKernelPrintf", 0x13A5ABEF, since = 150) { sceKernelPrintf(it) }
		registerFunctionRaw("sceKernelQueryMemoryInfo", 0x2A3E5280, since = 150) { sceKernelQueryMemoryInfo(it) }
		registerFunctionRaw("SysMemUserForUser_315AD3A0", 0x315AD3A0, since = 150) { SysMemUserForUser_315AD3A0(it) }
		registerFunctionRaw("SysMemUserForUser_342061E5", 0x342061E5, since = 150) { SysMemUserForUser_342061E5(it) }
		registerFunctionRaw("sceKernelDevkitVersion", 0x3FC9AE6A, since = 150) { sceKernelDevkitVersion(it) }
		registerFunctionRaw("SysMemUserForUser_50F61D8A", 0x50F61D8A, since = 150) { SysMemUserForUser_50F61D8A(it) }
		registerFunctionRaw("sceKernelSetCompiledSdkVersion", 0x7591C7DB, since = 150) { sceKernelSetCompiledSdkVersion(it) }
		registerFunctionRaw("SysMemUserForUser_91DE343C", 0x91DE343C, since = 150) { SysMemUserForUser_91DE343C(it) }
		registerFunctionRaw("sceKernelMaxFreeMemSize", 0xA291F107, since = 150) { sceKernelMaxFreeMemSize(it) }
		registerFunctionRaw("SysMemUserForUser_A6848DF8", 0xA6848DF8, since = 150) { SysMemUserForUser_A6848DF8(it) }
		registerFunctionRaw("SysMemUserForUser_ACBD88CA", 0xACBD88CA, since = 150) { SysMemUserForUser_ACBD88CA(it) }
		registerFunctionRaw("sceKernelFreePartitionMemory", 0xB6D61D02, since = 150) { sceKernelFreePartitionMemory(it) }
		registerFunctionRaw("SysMemUserForUser_D8DE5C1E", 0xD8DE5C1E, since = 150) { SysMemUserForUser_D8DE5C1E(it) }
		registerFunctionRaw("SysMemUserForUser_DB83A952", 0xDB83A952, since = 150) { SysMemUserForUser_DB83A952(it) }
		registerFunctionRaw("SysMemUserForUser_EBD5C3E6", 0xEBD5C3E6, since = 150) { SysMemUserForUser_EBD5C3E6(it) }
		registerFunctionRaw("sceKernelSetCompilerVersion", 0xF77D77CB, since = 150) { sceKernelSetCompilerVersion(it) }
		registerFunctionRaw("sceKernelTotalFreeMemSize", 0xF919F628, since = 150) { sceKernelTotalFreeMemSize(it) }
		registerFunctionRaw("sceKernelGetCompiledSdkVersion", 0xFC114573, since = 150) { sceKernelGetCompiledSdkVersion(it) }
		registerFunctionRaw("SysMemUserForUser_FE707FDF", 0xFE707FDF, since = 150) { SysMemUserForUser_FE707FDF(it) }
	}
}