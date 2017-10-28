package com.soywiz.kpspemu.cpu.interpreter

import com.soywiz.korio.lang.format
import com.soywiz.kpspemu.cpu.CpuState
import com.soywiz.kpspemu.cpu.InstructionDecoder
import com.soywiz.kpspemu.cpu.InstructionEvaluator
import com.soywiz.kpspemu.cpu.InstructionType

object InstructionInterpreter : InstructionEvaluator<CpuState>(), InstructionDecoder {
	override fun unimplemented(s: CpuState, i: InstructionType): Unit = TODO("unimplemented: ${i.name} : " + i + " at ${"%08X".format(s._PC)}")

	override fun lui(s: CpuState) = s { RT = (U_IMM16 shl 16) }
	override fun addu(s: CpuState) = s { RD = RS + RT }
	override fun subu(s: CpuState) = s { RD = RS - RT }
	override fun addiu(s: CpuState) = s { RT = RS + U_IMM16 }
	override fun ori(s: CpuState) = s { RT = RS or S_IMM16 }
	override fun sll(s: CpuState) = s { RD = RT shl POS }
	override fun sb(s: CpuState) = s { mem.sb(RS + S_IMM16, RT) }
	override fun syscall(s: CpuState) = s {
		this.syscall(SYSCALL)
	}
}

