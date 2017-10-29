package com.soywiz.kpspemu.format.elf

import com.soywiz.korio.stream.*
import com.soywiz.korio.util.extract
import com.soywiz.korio.util.insert
import com.soywiz.kpspemu.cpu.CpuState
import com.soywiz.kpspemu.mem.Memory

data class ElfPspModuleInfo(
	val moduleAtributes: Int,
	val moduleVersion: Int,
	val name: String,
	val gp: Int,
	val exportsStart: Int,
	val exportsEnd: Int,
	val importsStart: Int,
	val importsEnd: Int
) {
	var pc: Int = 0

	// http://hitmen.c02.at/files/yapspd/psp_doc/chap26.html
	// 26.2.2.8
	companion object {
		operator fun invoke(s: SyncStream) = s.run {
			ElfPspModuleInfo(
				moduleAtributes = readU16_le(),
				moduleVersion = readU16_le(),
				name = readStringz(28),
				gp = readS32_le(),
				exportsStart = readS32_le(),
				exportsEnd = readS32_le(),
				importsStart = readS32_le(),
				importsEnd = readS32_le()
			)
		}
	}
}

data class ElfPspModuleImport(
	val nameOffset: Int,
	val version: Int,
	val flags: Int,
	val entrySize: Int,
	val functionCount: Int,
	val variableCount: Int,
	val nidAddress: Int,
	val callAddress: Int
) {
	var name: String = ""

	companion object {
		val SIZE = 20

		operator fun invoke(s: SyncStream) = s.run {
			ElfPspModuleImport(
				nameOffset = readS32_le(),
				version = readU16_le(),
				flags = readU16_le(),
				entrySize = readU8(),
				variableCount = readU8(),
				functionCount = readU16_le(),
				nidAddress = readS32_le(),
				callAddress = readS32_le()
			)
		}
	}
}

class ElfPspModuleExport(
	var name: Int,
	var version: Int,
	var flags: Int,
	var entrySize: Int,
	var variableCount: Int,
	var functionCount: Int,
	var exports: Int
) {
	companion object {
		operator fun invoke(s: SyncStream) = s.run {
			ElfPspModuleExport(
				name = readS32_le(),
				version = readU16_le(),
				flags = readU16_le(),
				entrySize = readU8(),
				variableCount = readU8(),
				functionCount = readU16_le(),
				exports = readS32_le()
			)
		}
	}
}

data class ElfPspModuleInfoAtributesEnum(val id: Int) {
	companion object {
		val UserMode = 0x0000
		val KernelMode = 0x100
	}
}

data class Instruction(val address: Int, var data: Int) {
	var u_imm16: Int get() = data.extract(0, 16); set(value) = run { data = data.insert(value, 0, 16) }
}

class NativeFunction {
	var name: String = ""
	var nid: Int = 0
	var firmwareVersion: Int = 150
	var nativeCall: () -> Unit = {}
	var call: (Int, CpuState) -> Unit = { a, b -> }
}

class InstructionReader(
	private val memory: Memory
) {
	fun read(address: Int): Instruction = Instruction(address, memory.lw(address))
	fun write(address: Int, instruction: Instruction) = memory.sw(address, instruction.data)
}

/*
class PspElfLoader(
	private var memory: Memory,
	private var memoryManager: MemoryManager,
	private var moduleManager: ModuleManager,
	private var syscallManager: SyscallManager
) {
	lateinit private var elfLoader: Elf
	var moduleInfo: ElfPspModuleInfo
	var assembler = MipsAssembler()
	var baseAddress: Int = 0
	var partition: _manager.MemoryPartition
	var elfDwarfLoader: ElfDwarfLoader

	fun load(stream: SyncStream) {
		//console.warn('PspElfLoader.load');
		this.elfLoader = Elf.read(stream)

		//ElfSectionHeaderFlags.Allocate

		this.allocateMemory()
		this.writeToMemory()
		this.relocateFromHeaders()
		this.readModuleInfo()
		this.updateModuleImports()

		this.elfDwarfLoader = ElfDwarfLoader()
		this.elfDwarfLoader.parseElfLoader(this.elfLoader)

		//this.memory.dump(); debugger;

		//this.elfDwarfLoader.getSymbolAt();

		//logger.log(this.moduleInfo);
	}

	fun getSymbolAt(address: Int) {
		return this.elfDwarfLoader.getSymbolAt(address)
	}

	private fun getSectionHeaderMemoryStream(sectionHeader: ElfSectionHeader): SyncStream {
		return this.memory.getPointerStream(this.baseAddress + sectionHeader.address, sectionHeader.size)
	}

	private fun readModuleInfo() {
		this.moduleInfo = ElfPspModuleInfo(this.getSectionHeaderMemoryStream(this.elfLoader.getSectionHeader(".rodata.sceModuleInfo")))
		this.moduleInfo.pc = this.baseAddress + this.elfLoader.header.entryPoint
	}

	private fun allocateMemory() {
		this.baseAddress = 0

		if (this.elfLoader.needsRelocation) {
			this.baseAddress = this.memoryManager.userPartition.childPartitions.sortBy(partition => partition . size).reverse().first().low
			this.baseAddress = this.baseAddress.nextAlignedTo(0x1000)
			//this.baseAddress = 0x08800000 + 0x4000;

		}

		var lowest = 0xFFFFFFFF.toInt()
		var highest = 0
		for (section in this.elfLoader.sectionHeaders.filter { it.flags hasFlag ElfSectionHeaderFlags.Allocate }) {
			lowest = min(lowest, (this.baseAddress + section.address))
			highest = max(highest, (this.baseAddress + section.address + section.size))
		}

		for (program in this.elfLoader.programHeaders) {
			lowest = min(lowest, (this.baseAddress + program.virtualAddress))
			highest = max(highest, (this.baseAddress + program.virtualAddress + program.memorySize))
		}

		var memorySegment = this.memoryManager.userPartition.allocateSet(highest - lowest, lowest, 'Elf')
	}

	private fun relocateFromHeaders() {
		var RelocProgramIndex = 0
		for (programHeader in this.elfLoader.programHeaders) {
			when (programHeader.type) {
				ElfProgramHeaderType.Reloc1 -> {
					println("SKIPPING Elf.ProgramHeader.TypeEnum.Reloc1!")
				}
				ElfProgramHeaderType.Reloc2 -> {
					throw Exception("Not implemented")
				}
			}
		}

		var RelocSectionIndex = 0
		for (sectionHeader in elfLoader.sectionHeaders) {
			//RelocOutput.WriteLine("Section Header: %d : %s".Sprintf(RelocSectionIndex++, SectionHeader.ToString()));
			//println(sprintf('Section Header: '));

			when (sectionHeader.type) {
				ElfSectionHeaderType.Relocation -> {
					println(sectionHeader)
					println("Not implemented ElfSectionHeaderType.Relocation")
				}

				ElfSectionHeaderType.PrxRelocation -> {
					val relocs = (0 until sectionHeader.stream.length / ElfReloc.SIZE).map { ElfReloc(sectionHeader.stream) }
					this.relocateRelocs(relocs)
				}

				ElfSectionHeaderType.PrxRelocation_FW5 -> throw Error("Not implemented ElfSectionHeader.Type.PrxRelocation_FW5")
			}
		}
	}

	private fun relocateRelocs(relocs: List<ElfReloc>) {
		val baseAddress = this.baseAddress
		var hiValue: Int = 0
		var deferredHi16 = arrayListOf<Int>()
		val instructionReader = InstructionReader(this.memory)

		for (index in 0 until relocs.size) {
			val reloc = relocs[index]
			if (reloc.type == ElfRelocType.StopRelocation) break

			val pointerBaseOffset = this.elfLoader.programHeaders[reloc.pointerSectionHeaderBase].virtualAddress
			val pointeeBaseOffset = this.elfLoader.programHeaders[reloc.pointeeSectionHeaderBase].virtualAddress

			// Address of data to relocate
			val RelocatedPointerAddress = (baseAddress + reloc.pointerAddress + pointerBaseOffset)

			// Value of data to relocate
			val instruction = instructionReader.read(RelocatedPointerAddress)

			val S = baseAddress + pointeeBaseOffset
			val GP_ADDR = (baseAddress + reloc.pointerAddress)
			var GP_OFFSET = GP_ADDR - (baseAddress and 0xFFFF0000.toInt())

			when (reloc.type) {
				ElfRelocType.None -> Unit
				ElfRelocType.Mips16 -> instruction.u_imm16 += S
				ElfRelocType.Mips32 -> instruction.data += S
				ElfRelocType.MipsRel32 -> throw Exception("Not implemented MipsRel32")
				ElfRelocType.Mips26 -> instruction.jump_real = instruction.jump_real + S
				ElfRelocType.MipsHi16 -> {
					hiValue = instruction.u_imm16
					deferredHi16.add(RelocatedPointerAddress)
				}
				ElfRelocType.MipsLo16 -> {
					val A = instruction.u_imm16

					instruction.u_imm16 = ((hiValue shl 16) or (A and 0x0000FFFF)) + S

					for (data_addr2 in deferredHi16) {
						val data2 = instructionReader.read(data_addr2)
						var result = ((data2.data and 0x0000FFFF) shl 16) + A + S
						if ((A and 0x8000) != 0) {
							result -= 0x10000
						}
						if ((result and 0x8000) != 0) {
							result += 0x10000
						}
						data2.u_imm16 = (result ushr 16)
						instructionReader.write(data_addr2, data2)
					}

					deferredHi16 = arrayListOf()
				}
				ElfRelocType.MipsGpRel16 -> Unit
				else -> throw (Error("RelocType %d not implemented".format(reloc.type)))
			}

			instructionReader.write(RelocatedPointerAddress, instruction)
		}
	}

	private fun writeToMemory() {
		val needsRelocate = this.elfLoader.needsRelocation

		//var loadAddress = this.elfLoader.programHeaders[0].psysicalAddress;
		val loadAddress = this.baseAddress

		println("PspElfLoader: needsRelocate=%s, loadAddress=%08X".format(needsRelocate, loadAddress))
		//console.log(moduleInfo);

		for (programHeader in this.elfLoader.programHeaders.filter { it.type == ElfProgramHeaderType.Load }) {
			val fileOffset = programHeader.offset
			val memOffset = this.baseAddress + programHeader.virtualAddress
			val fileSize = programHeader.fileSize
			val memSize = programHeader.memorySize

			this.elfLoader.stream.sliceWithSize(fileOffset, fileSize).copyTo(this.memory.getPointerStream(memOffset, fileSize))
			this.memory.memset(memOffset + fileSize, 0, memSize - fileSize)

			//this.getSectionHeaderMemoryStream
			println("Program Header: " + "%08X:%08X, %08X:%08X".format(fileOffset, fileSize, memOffset, memSize))
		}

		for (sectionHeader in this.elfLoader.sectionHeaders.filter { it.flags hasFlag ElfSectionHeaderFlags.Allocate }) {
			val low = loadAddress + sectionHeader.address

			println("Section Header: %s LOW:%08X, SIZE:%08X".format(sectionHeader.toString(), low, sectionHeader.size))

			//console.log(sectionHeader);
			when (sectionHeader.type) {
				ElfSectionHeaderType.NoBits -> {
					for (n in 0 until sectionHeader.size) this.memory.sb(low + n, 0)
				}
				ElfSectionHeaderType.ProgramBits -> {
					val stream = sectionHeader.stream

					var length = stream.length

					//console.log(sprintf('low: %08X, %08X, size: %08X', sectionHeader.address, low, stream.length));
					this.memory.write(low, stream.readAll())

				}
				else -> {
					//console.log(sprintf('low: %08X type: %08X', low, sectionHeader.type));
				}
			}
		}

	}

	private fun updateModuleImports() {
		val moduleInfo = this.moduleInfo
		println(moduleInfo)
		val importsBytesSize = moduleInfo.importsEnd - moduleInfo.importsStart
		val importsStream = this.memory.openSync().slice(moduleInfo.importsStart until moduleInfo.importsEnd)
		val importsCount = importsBytesSize / ElfPspModuleImport.SIZE
		val imports = (0 until importsCount).map { ElfPspModuleImport(importsStream) }
		for (_import in imports) {
			_import.name = this.memory.readStringz(_import.nameOffset)
			val imported = this.updateModuleFunctions(_import)
			this.updateModuleVars(_import)
			println("Imported: " + imported.name + " " + imported.registeredNativeFunctions.map { it.name })
		}
		//console.log(imports);
	}

	private fun updateModuleFunctions(moduleImport: ElfPspModuleImport): registeredNativeFunctions {
		val _module = this.moduleManager.getByName(moduleImport.name)
		val nidsStream = this.memory.openSync().sliceWithSize(moduleImport.nidAddress, moduleImport.functionCount * 4)
		val callStream = this.memory.openSync().sliceWithSize(moduleImport.callAddress, moduleImport.functionCount * 8)
		val registeredNativeFunctions = arrayListOf<NativeFunction>()
		val unknownFunctions = arrayListOf<String>()

		val registerN = { nid: Int, n: Int ->
			var nfunc: NativeFunction
			nfunc = _module.getByNid(nid)

			if (nfunc != null) {
				unknownFunctions.add("'%s':0x%08X".format(_module.moduleName, nid))

				nfunc = NativeFunction()
				nfunc.name = "%s:0x%08X".format(moduleImport.name, nid)
				nfunc.nid = nid
				nfunc.firmwareVersion = 150
				nfunc.nativeCall = {
					println(_module)
					println("updateModuleFunctions: Not implemented '" + nfunc.name + "'")
					Debugger.enterDebugger()
					throw (Error("updateModuleFunctions: Not implemented '" + nfunc.name + "'"))
				}
				nfunc.call = { context, state ->
					nfunc.nativeCall()
				}
			}

			registeredNativeFunctions.add(nfunc)

			val syscallId = this.syscallManager.register(nfunc)
			//printf("%s:%08X -> %s", moduleImport.name, nid, syscallId);
			return syscallId
		}

		for (n in 0 until moduleImport.functionCount) {
			val nid = nidsStream.readS32_le()
			val syscall = registerN(nid, n)

			callStream.write32_le(this.assembler.assemble(0, "jr $31")[0].data)
			callStream.write32_le(this.assembler.assemble(0, "syscall %d".format(syscall))[0].data)
		}

		if (unknownFunctions.size > 0) {
			println("Can't find functions: " + unknownFunctions)
		}

		return Res1(name = moduleImport.name, registeredNativeFunctions = registeredNativeFunctions)
	}

	class Res1(val name: String, val registeredNativeFunctions: NativeFunction)

	private fun updateModuleVars(moduleImport: ElfPspModuleImport) {
	}
}
*/
