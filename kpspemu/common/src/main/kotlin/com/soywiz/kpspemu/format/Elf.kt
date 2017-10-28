package com.soywiz.kpspemu.format

import com.soywiz.korio.error.invalidOp
import com.soywiz.korio.stream.*
import com.soywiz.korio.util.UByteArray

class Elf {
	fun read(stream: SyncStream) {
		val header = Header(stream)
		println(header)

		var programHeadersStream = stream.sliceWithSize(header.programHeaderOffset.toLong(), (header.programHeaderCount * header.programHeaderEntrySize).toLong());
		var sectionHeadersStream = stream.sliceWithSize(header.sectionHeaderOffset.toLong(), (header.sectionHeaderCount * header.sectionHeaderEntrySize).toLong());

		println("test")

		//this.programHeaders = StructArray<ElfProgramHeader>(ElfProgramHeader.struct, this.header.programHeaderCount).read(programHeadersStream);
		//this.sectionHeaders = StructArray<ElfSectionHeader>(ElfSectionHeader.struct, this.header.sectionHeaderCount).read(sectionHeadersStream);
//
		//this.sectionHeaderStringTable = this.sectionHeaders[this.header.sectionHeaderStringTable];
		//this.stringTableStream = this.getSectionHeaderFileStream(this.sectionHeaderStringTable);
//
		//this.sectionHeadersByName = {};
		//this.sectionHeaders.forEach((sectionHeader) => {
		//	var name = this.getStringFromStringTable(sectionHeader.nameOffset);
		//	sectionHeader.name = name;
		//	if (sectionHeader.type != ElfSectionHeaderType.Null) {
		//		sectionHeader.stream = this.getSectionHeaderFileStream(sectionHeader);
		//	}
		//	this.sectionHeadersByName[name] = sectionHeader;
		//});


	}

	data class Header(
		val magic: String,
		val clazz: Int,
		val data: Int,
		val idVersion: Int,
		val padding: UByteArray,
		val type: Int,
		val machine: Int,
		val version: Int,
		val entryPoint: Int,
		val programHeaderOffset: Int,
		val sectionHeaderOffset: Int,
		val flags: Int,
		val elfHeaderSize: Int,
		val programHeaderEntrySize: Int,
		val programHeaderCount: Int,
		val sectionHeaderEntrySize: Int,
		val sectionHeaderCount: Int,
		val sectionHeaderStringTable: Int
	) {
		companion object {
			operator fun invoke(s: SyncStream): Header = s.run {
				return Header(
					magic = readStringz(4).apply { if (this != "\u007FELF") invalidOp("Not an ELF file") },
					clazz = readU8(),
					data = readU8(),
					idVersion = readU8(),
					padding = readUByteArray(9),
					type = readU16_le(),
					machine = readU16_le(),
					version = readS32_le(),
					entryPoint = readS32_le(),
					programHeaderOffset = readS32_le(),
					sectionHeaderOffset = readS32_le(),
					flags = readS32_le(),
					elfHeaderSize = readS16_le(),
					programHeaderEntrySize = readS16_le(),
					programHeaderCount = readU16_le(),
					sectionHeaderEntrySize = readU16_le(),
					sectionHeaderCount = readU16_le(),
					sectionHeaderStringTable = readU16_le()
				)
			}
		}

		val hasValidMagic: Boolean get() = this.magic == "\u007FELF";
		val hasValidMachine: Boolean get() = this.machine == ElfMachine.ALLEGREX.id
		val hasValidType: Boolean get() = listOf(ElfType.Executable.id, ElfType.Prx.id).contains(this.type)

		init {
			if (!hasValidMagic) invalidOp("Not an ELF file");
			if (!hasValidMachine) invalidOp("Not a PSP ELF file");
			if (!hasValidType) invalidOp("Not a executable or a Prx but has type $type");
		}
	}
}

open class BaseEnum<T : BaseEnum.Id>(val values: Array<T>) {
	interface Id {
		val id: Int
	}

	val BY_ID = values.map { it.id to it }.toMap()
	operator fun get(index: Int) = BY_ID[index]!!
}

enum class ElfProgramHeaderType(override val id: Int) : BaseEnum.Id {
	NoLoad(0),
	Load(1),
	Reloc1(0x700000A0),
	Reloc2(0x700000A1);

	companion object : BaseEnum<ElfProgramHeaderType>(values())
}

enum class ElfSectionHeaderType(override val id: Int) : BaseEnum.Id {
	Null(0),
	ProgramBits(1),
	SYMTAB(2),
	STRTAB(3),
	RELA(4),
	HASH(5),
	DYNAMIC(6),
	NOTE(7),
	NoBits(8),
	Relocation(9),
	SHLIB(10),
	DYNSYM(11),

	LOPROC(0x70000000), HIPROC(0x7FFFFFFF),
	LOUSER(0x80000000.toInt()), HIUSER(0xFFFFFFFF.toInt()),

	PrxRelocation(LOPROC.id or 0xA0),
	PrxRelocation_FW5(LOPROC.id or 0xA1);

	companion object : BaseEnum<ElfSectionHeaderType>(values())
}

enum class ElfSectionHeaderFlags(override val id: Int) : BaseEnum.Id {
	None(0),
	Write(1),
	Allocate(2),
	Execute(4);

	companion object : BaseEnum<ElfSectionHeaderFlags>(values())
}

enum class ElfProgramHeaderFlags(override val id: Int) : BaseEnum.Id {
	Executable(0x1),
	// Note: demo PRX's were found to be not writable
	Writable(0x2),
	Readable(0x4);

	companion object : BaseEnum<ElfProgramHeaderFlags>(values())
}

enum class ElfType(override val id: Int) : BaseEnum.Id {
	Executable(0x0002),
	Prx(0xFFA0);

	companion object : BaseEnum<ElfType>(values())
}

enum class ElfMachine(override val id: Int) : BaseEnum.Id {
	ALLEGREX(8);

	companion object : BaseEnum<ElfMachine>(values())
}

enum class ElfPspModuleFlags(override val id: Int) : BaseEnum.Id { // ushort
	User(0x0000),
	Kernel(0x1000);

	companion object : BaseEnum<ElfPspModuleFlags>(values())
}

enum class ElfPspLibFlags(override val id: Int) : BaseEnum.Id { // ushort
	DirectJump(0x0001),
	Syscall(0x4000),
	SysLib(0x8000);

	companion object : BaseEnum<ElfPspLibFlags>(values())
}

enum class ElfPspModuleNids(override val id: Int) : BaseEnum.Id {  // uint
	MODULE_INFO(0xF01D73A7.toInt()),
	MODULE_BOOTSTART(0xD3744BE0.toInt()),
	MODULE_REBOOT_BEFORE(0x2F064FA6),
	MODULE_START(0xD632ACDB.toInt()),
	MODULE_START_THREAD_PARAMETER(0x0F7C276C),
	MODULE_STOP(0xCEE8593C.toInt()),
	MODULE_STOP_THREAD_PARAMETER(0xCF0CC697.toInt());

	companion object : BaseEnum<ElfPspModuleNids>(values())
}


enum class ElfRelocType(override val id: Int) : BaseEnum.Id {
	None(0),
	Mips16(1),
	Mips32(2),
	MipsRel32(3),
	Mips26(4),
	MipsHi16(5),
	MipsLo16(6),
	MipsGpRel16(7),
	MipsLiteral(8),
	MipsGot16(9),
	MipsPc16(10),
	MipsCall16(11),
	MipsGpRel32(12),
	StopRelocation(0xFF);

	companion object : BaseEnum<ElfRelocType>(values())
}

/*
class ElfReloc {
	pointerAddress: number;
	info: number;

	get pointeeSectionHeaderBase() { return (this.info >> 16) & 0xFF; }
	get pointerSectionHeaderBase() { return (this.info >> 8) & 0xFF; }
	get type() { return <ElfRelocType>((this.info >> 0) & 0xFF); }

	static struct = StructClass.create<ElfReloc>(ElfReloc, [
	{ pointerAddress: UInt32 },
	{ info: UInt32 },
	]);
}
*/

data class ElfProgramHeader(
	val type: ElfProgramHeaderType,
	val offset: Int,
	val virtualAddress: Int,
	val psysicalAddress: Int,
	val fileSize: Int,
	val memorySize: Int,
	val flags: ElfProgramHeaderFlags,
	val alignment: Int
) {
	companion object {
		operator fun invoke(s: SyncStream): ElfProgramHeader = s.run {
			ElfProgramHeader(
				type = ElfProgramHeaderType[readS32_le()],
				offset = readS32_le(),
				virtualAddress = readS32_le(),
				psysicalAddress = readS32_le(),
				fileSize = readS32_le(),
				memorySize = readS32_le(),
				flags = ElfProgramHeaderFlags[readS32_le()],
				alignment = readS32_le()
			)
		}
	}
}

data class ElfSectionHeader(
	val nameOffset: Int,
	val name: String,
	val type: ElfSectionHeaderType,
	val flags: ElfSectionHeaderFlags,
	val address: Int,
	val offset: Int,
	val size: Int,
	val link: Int,
	val info: Int,
	val addressAlign: Int,
	val entitySize: Int
) {
	lateinit var stream: SyncStream

	companion object {
		operator fun invoke(s: SyncStream): ElfSectionHeader = s.run {
			ElfSectionHeader(
				nameOffset = s.readS32_le(),
				name = "...",
				type = ElfSectionHeaderType[s.readS32_le()],
				flags = ElfSectionHeaderFlags[s.readS32_le()],
				address = s.readS32_le(),
				offset = s.readS32_le(),
				size = s.readS32_le(),
				link = s.readS32_le(),
				info = s.readS32_le(),
				addressAlign = s.readS32_le(),
				entitySize = s.readS32_le()
			)
		}
	}
}