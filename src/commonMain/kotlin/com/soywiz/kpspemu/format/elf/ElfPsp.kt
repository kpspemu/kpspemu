package com.soywiz.kpspemu.format.elf

import com.soywiz.klogger.*
import com.soywiz.kmem.*
import com.soywiz.korio.lang.*
import com.soywiz.korio.stream.*
import com.soywiz.korio.util.*
import com.soywiz.kpspemu.*
import com.soywiz.kpspemu.cpu.*
import com.soywiz.kpspemu.hle.*
import com.soywiz.kpspemu.hle.manager.*
import com.soywiz.kpspemu.mem.*
import kotlin.collections.set
import kotlin.math.*
import kotlin.reflect.*

data class ElfPspModuleInfo(
    val moduleAtributes: Int,
    val moduleVersion: Int,
    val name: String,
    val GP: Int,
    val exportsStart: Int,
    val exportsEnd: Int,
    val importsStart: Int,
    val importsEnd: Int,
    var PC: Int = 0
) {

    // http://hitmen.c02.at/files/yapspd/psp_doc/chap26.html
    // 26.2.2.8
    companion object {
        operator fun invoke(s: SyncStream) = s.run {
            ElfPspModuleInfo(
                moduleAtributes = readU16_le(),
                moduleVersion = readU16_le(),
                name = readStringz(28),
                GP = readS32_le(),
                exportsStart = readS32_le(),
                exportsEnd = readS32_le(),
                importsStart = readS32_le(),
                importsEnd = readS32_le(),
                PC = 0
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

class InstructionReader(
    private val memory: Memory
) {
    fun read(address: Int): Instruction = Instruction(address, memory.lw(address))
    fun write(address: Int, instruction: Instruction) = memory.sw(address, instruction.data)
}

fun Emulator.loadElf(file: SyncStream): PspElf = PspElf.loadInto(file, this)

fun Emulator.loadElfAndSetRegisters(file: SyncStream, args: List<String>): PspElf {
    logger.warn { "loadElfAndSetRegisters: $args" }
    val elf = loadElf(file)
    val thread = threadManager.create("_start", 0, 0, 0x1000, 0, mem.ptr(0))
    val data = thread.putDataInStack(args.map { it.toByteArray(UTF8) + byteArrayOf(0) }.join())
    val state = thread.state
    state.setPC(elf.moduleInfo.PC)
    state.GP = elf.moduleInfo.GP
    state.r4 = data.size
    state.r5 = data.addr
    thread.start()
    return elf
}

class PspElf private constructor(
    private val memory: Memory,
    private val memoryManager: MemoryManager,
    private val moduleManager: ModuleManager,
    private val syscallManager: SyscallManager,
    private val addressInfo: AddressInfo
) {
    private val logger = Logger("ElfPsp")

    lateinit var elf: Elf; private set
    lateinit var moduleInfo: ElfPspModuleInfo; private set
    lateinit var dwarf: ElfDwarf; private set
    var baseAddress: Int = 0; private set

    companion object {
        fun loadInto(stream: SyncStream, emulator: Emulator): PspElf {
            val loader = PspElf(
                emulator.mem,
                emulator.memoryManager,
                emulator.moduleManager,
                emulator.syscalls,
                emulator.nameProvider
            )
            loader.load(stream)
            return loader
        }
    }

    fun load(stream: SyncStream) {
        //console.warn('PspElfLoader.load');
        this.elf = Elf.read(stream)

        //ElfSectionHeaderFlags.Allocate

        this.allocateMemory()
        this.writeToMemory()
        this.relocateFromHeaders()
        this.readModuleInfo()
        this.updateModuleImports()

        this.dwarf = ElfDwarf()
        this.dwarf.parseElfLoader(this.elf)

        //this.memory.dump(); debugger;

        //this.elfDwarfLoader.getSymbolAt();

        //logger.log(this.moduleInfo);
    }

    fun getSymbolAt(address: Int) = this.dwarf.getSymbolAt(address)

    private fun getSectionHeaderMemoryStream(sectionHeader: ElfSectionHeader): SyncStream {
        return this.memory.getPointerStream(this.baseAddress + sectionHeader.address, sectionHeader.size)
    }

    private fun readModuleInfo() {
        this.moduleInfo =
                ElfPspModuleInfo(this.getSectionHeaderMemoryStream(this.elf.getSectionHeader(".rodata.sceModuleInfo")))
        this.moduleInfo.PC = this.baseAddress + this.elf.header.entryPoint
    }

    private fun allocateMemory() {
        baseAddress = 0

        if (elf.needsRelocation) {
            baseAddress = memoryManager.userPartition.childPartitions.sortedBy { it.size }.last().low.toInt()
            baseAddress = baseAddress.nextAlignedTo(0x1000) + 0x4000
            //this.baseAddress = 0x08800000 + 0x4000;
        }

        var lowest: Long = 0xFFFFFFFF
        var highest: Long = 0
        for (section in this.elf.sectionHeaders.filter { it.flags hasFlag ElfSectionHeaderFlags.Allocate }) {
            lowest = min(lowest, (this.baseAddress.toLong() + section.address))
            highest = max(highest, (this.baseAddress.toLong() + section.address + section.size))
        }

        for (program in this.elf.programHeaders) {
            lowest = min(lowest, (this.baseAddress.toLong() + program.virtualAddress))
            highest = max(highest, (this.baseAddress.toLong() + program.virtualAddress + program.memorySize))
        }

        var memorySegment = this.memoryManager.userPartition.allocateSet(highest - lowest, lowest, "Elf")
    }

    private fun relocateFromHeaders() {
        var RelocProgramIndex = 0
        for (programHeader in this.elf.programHeaders) {
            when (programHeader.type) {
                ElfProgramHeaderType.Reloc1 -> {
                    logger.info { "SKIPPING Elf.ProgramHeader.TypeEnum.Reloc1!" }
                }
                ElfProgramHeaderType.Reloc2 -> {
                    throw Exception("Not implemented")
                }
            }
        }

        var RelocSectionIndex = 0
        for (sectionHeader in elf.sectionHeaders) {
            //RelocOutput.WriteLine("Section Header: %d : %s".Sprintf(RelocSectionIndex++, SectionHeader.ToString()));
            //println(sprintf('Section Header: '));

            when (sectionHeader.type) {
                ElfSectionHeaderType.Relocation -> {
                    logger.error { "sectionHeader: $sectionHeader" }
                    logger.error { "Not implemented ElfSectionHeaderType.Relocation" }
                }

                ElfSectionHeaderType.PrxRelocation -> {
                    val relocs =
                        (0 until sectionHeader.stream.length / ElfReloc.SIZE).map { ElfReloc(sectionHeader.stream) }
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

            val pointerBaseOffset = this.elf.programHeaders[reloc.pointerSectionHeaderBase].virtualAddress
            val pointeeBaseOffset = this.elf.programHeaders[reloc.pointeeSectionHeaderBase].virtualAddress

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
        val needsRelocate = this.elf.needsRelocation

        //var loadAddress = this.elfLoader.programHeaders[0].psysicalAddress;
        val loadAddress = this.baseAddress

        logger.info { "PspElfLoader: needsRelocate=%s, loadAddress=%08X".format(needsRelocate, loadAddress) }
        //console.log(moduleInfo);

        for (programHeader in this.elf.programHeaders.filter { it.type == ElfProgramHeaderType.Load }) {
            val fileOffset = programHeader.offset
            val memOffset = this.baseAddress + programHeader.virtualAddress
            val fileSize = programHeader.fileSize
            val memSize = programHeader.memorySize

            this.elf.stream.sliceWithSize(fileOffset, fileSize)
                .copyTo(this.memory.getPointerStream(memOffset, fileSize))
            this.memory.memset(memOffset + fileSize, 0, memSize - fileSize)

            //this.getSectionHeaderMemoryStream
            logger.info { "Program Header: " + "%08X:%08X, %08X:%08X".format(fileOffset, fileSize, memOffset, memSize) }
        }

        for (sectionHeader in this.elf.sectionHeaders.filter { it.flags hasFlag ElfSectionHeaderFlags.Allocate }) {
            val low = loadAddress + sectionHeader.address

            logger.info {
                "Section Header: %s LOW:%08X, SIZE:%08X".format(
                    sectionHeader.toString(),
                    low,
                    sectionHeader.size
                )
            }

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
        logger.info { "updateModuleImports.moduleInfo: $moduleInfo" }
        val importsBytesSize = moduleInfo.importsEnd - moduleInfo.importsStart
        val importsStream = this.memory.openSync().slice(moduleInfo.importsStart until moduleInfo.importsEnd)
        val importsCount = importsBytesSize / ElfPspModuleImport.SIZE
        val imports = (0 until importsCount).map { ElfPspModuleImport(importsStream) }
        for (_import in imports) {
            _import.name = this.memory.readStringz(_import.nameOffset)
            val imported = this.updateModuleFunctions(_import)
            this.updateModuleVars(_import)
            logger.info { "Imported: ${imported.name} ${imported.registeredNativeFunctions.map { it.name }}" }
        }
        //console.log(imports);
    }

    private fun updateModuleFunctions(moduleImport: ElfPspModuleImport): Res1 {
        logger.info { "Import module: ${moduleImport.name}" }
        val _module = this.moduleManager.getByName(moduleImport.name)
        val nidsStream = this.memory.openSync().sliceWithSize(moduleImport.nidAddress, moduleImport.functionCount * 4)
        val callStream = this.memory.openSync().sliceWithSize(moduleImport.callAddress, moduleImport.functionCount * 8)
        val registeredNativeFunctions = arrayListOf<NativeFunction>()
        val unknownFunctions = arrayListOf<String>()

        val registerN: (Int, Int) -> Int = { nid: Int, n: Int ->
            var nfunc: NativeFunction? = _module.getByNidOrNull(nid)

            if (nfunc == null) {
                unknownFunctions.add("'%s':0x%08X".format(_module.name, nid))

                nfunc = NativeFunction(
                    name = "%s:0x%08X".format(moduleImport.name, nid),
                    nid = nid.toLong(),
                    since = 150,
                    syscall = -1,
                    function = { state ->
                        logger.error { "$_module" }
                        logger.error { "updateModuleFunctions: Not implemented '${nfunc?.name}'" }
                        Debugger.enterDebugger()
                        throw Error("updateModuleFunctions: Not implemented '${nfunc?.name}'")
                    }
                )
            }

            registeredNativeFunctions.add(nfunc)

            addressInfo.names[moduleImport.callAddress + n * 8] = nfunc.name

            val syscallId = this.syscallManager.register(nfunc)
            //printf("%s:%08X -> %s", moduleImport.name, nid, syscallId);
            syscallId
        }

        for (n in 0 until moduleImport.functionCount) {
            val nid = nidsStream.readS32_le()
            val syscall = registerN(nid, n)

            //callStream.write32_le(this.assembler.assemble(0, "jr $31")[0].data)
            //callStream.write32_le(this.assembler.assemble(0, "syscall %d".format(syscall))[0].data)
            callStream.write32_le(0b000000_11111_00000_00000_00000_001000) // jr $31
            callStream.write32_le(0b000000_00000000000000000000_001100 or (syscall shl 6)) // syscall <syscall>
        }

        if (unknownFunctions.size > 0) {
            logger.warn { "Can't find functions: " + unknownFunctions }
        }

        return Res1(name = moduleImport.name, registeredNativeFunctions = registeredNativeFunctions)
    }

    class Res1(val name: String, val registeredNativeFunctions: List<NativeFunction>)

    private fun updateModuleVars(moduleImport: ElfPspModuleImport) {
    }
}

// @TODO: Ask if we can inline this so it has the best performance possible
class Bits(val offset: Int, val size: Int) {
    operator fun getValue(i: Instruction, p: KProperty<*>): Int = i.data.extract(offset, size)
    operator fun setValue(i: Instruction, p: KProperty<*>, value: Int) =
        run { i.data = i.data.insert(value, offset, size) }
}

data class Instruction(val address: Int, var data: Int) {
    var u_imm16: Int by Bits(0, 16)
    var jump_bits: Int by Bits(0, 26)
    var jump_real: Int get() = jump_bits * 4; set(value) = run { jump_bits = value / 4 }

    //var u_imm16: Int get() = data.extract(0, 16); set(value) = run { data = data.insert(value, 0, 16) }
    //get jump_bits() { return this.extract(0, 26); } set jump_bits(value: number) { this.insert(0, 26, value); }
    //get jump_real() { return (this.jump_bits * 4) >>> 0; } set jump_real(value: number) { this.jump_bits = (value / 4) >>> 0; }

}

/*
export class Instruction {
	constructor(public PC: number, public data: number) {
	}

	static fromMemoryAndPC(memory: Memory, PC: number) { return new Instruction(PC, memory.readInt32(PC)); }

	extract(offset: number, length: number) { return BitUtils.extract(this.data, offset, length); }
	extract_s(offset: number, length: number) { return BitUtils.extractSigned(this.data, offset, length); }
	insert(offset: number, length: number, value: number) { this.data = BitUtils.insert(this.data, offset, length, value); }

	get rd() { return this.extract(11 + 5 * 0, 5); } set rd(value: number) { this.insert(11 + 5 * 0, 5, value); }
	get rt() { return this.extract(11 + 5 * 1, 5); } set rt(value: number) { this.insert(11 + 5 * 1, 5, value); }
	get rs() { return this.extract(11 + 5 * 2, 5); } set rs(value: number) { this.insert(11 + 5 * 2, 5, value); }

	get fd() { return this.extract(6 + 5 * 0, 5); } set fd(value: number) { this.insert(6 + 5 * 0, 5, value); }
	get fs() { return this.extract(6 + 5 * 1, 5); } set fs(value: number) { this.insert(6 + 5 * 1, 5, value); }
	get ft() { return this.extract(6 + 5 * 2, 5); } set ft(value: number) { this.insert(6 + 5 * 2, 5, value); }

	get VD() { return this.extract(0, 7); } set VD(value: number) { this.insert(0, 7, value); }
	get VS() { return this.extract(8, 7); } set VS(value: number) { this.insert(8, 7, value); }
	get VT() { return this.extract(16, 7); } set VT(value: number) { this.insert(16, 7, value); }
	get VT5_1() { return this.VT5 | (this.VT1 << 5); } set VT5_1(value: number) { this.VT5 = value; this.VT1 = (value >>> 5); }
	get IMM14() { return this.extract_s(2, 14); } set IMM14(value: number) { this.insert(2, 14, value); }

	get ONE() { return this.extract(7, 1); } set ONE(value: number) { this.insert(7, 1, value); }
	get TWO() { return this.extract(15, 1); } set TWO(value: number) { this.insert(15, 1, value); }
	get ONE_TWO() { return (1 + 1 * this.ONE + 2 * this.TWO); } set ONE_TWO(value: number) { this.ONE = (((value - 1) >>> 0) & 1); this.TWO = (((value - 1) >>> 1) & 1); }


	get IMM8() { return this.extract(16, 8); } set IMM8(value: number) { this.insert(16, 8, value); }
	get IMM5() { return this.extract(16, 5); } set IMM5(value: number) { this.insert(16, 5, value); }
	get IMM3() { return this.extract(18, 3); } set IMM3(value: number) { this.insert(18, 3, value); }
	get IMM7() { return this.extract(0, 7); } set IMM7(value: number) { this.insert(0, 7, value); }
	get IMM4() { return this.extract(0, 4); } set IMM4(value: number) { this.insert(0, 4, value); }
	get VT1() { return this.extract(0, 1); } set VT1(value: number) { this.insert(0, 1, value); }
	get VT2() { return this.extract(0, 2); } set VT2(value: number) { this.insert(0, 2, value); }
	get VT5() { return this.extract(16, 5); } set VT5(value: number) { this.insert(16, 5, value); }
	get VT5_2() { return this.VT5 | (this.VT2 << 5); }
	get IMM_HF() { return HalfFloat.toFloat(this.imm16); }

	get pos() { return this.lsb; } set pos(value: number) { this.lsb = value; }
	get size_e() { return this.msb + 1; } set size_e(value: number) { this.msb = value - 1; }
	get size_i() { return this.msb - this.lsb + 1; } set size_i(value: number) { this.msb = this.lsb + value - 1; }

	get lsb() { return this.extract(6 + 5 * 0, 5); } set lsb(value: number) { this.insert(6 + 5 * 0, 5, value); }
	get msb() { return this.extract(6 + 5 * 1, 5); } set msb(value: number) { this.insert(6 + 5 * 1, 5, value); }
	get c1cr() { return this.extract(6 + 5 * 1, 5); } set c1cr(value: number) { this.insert(6 + 5 * 1, 5, value); }

	get syscall() { return this.extract(6, 20); } set syscall(value: number) { this.insert(6, 20, value); }

	get imm16() { var res = this.u_imm16; if (res & 0x8000) res |= 0xFFFF0000; return res; } set imm16(value: number) { this.insert(0, 16, value); }
	get u_imm16() { return this.extract(0, 16); } set u_imm16(value: number) { this.insert(0, 16, value); }
	get u_imm26() { return this.extract(0, 26); } set u_imm26(value: number) { this.insert(0, 26, value); }

	get jump_bits() { return this.extract(0, 26); } set jump_bits(value: number) { this.insert(0, 26, value); }
	get jump_real() { return (this.jump_bits * 4) >>> 0; } set jump_real(value: number) { this.jump_bits = (value / 4) >>> 0; }

	set branch_address(value:number) { this.imm16 = (value - this.PC - 4) / 4; }
	set jump_address(value:number) { this.u_imm26 = value / 4; }

	get branch_address() { return this.PC + this.imm16 * 4 + 4; }
	get jump_address() { return this.u_imm26 * 4; }
}
*/