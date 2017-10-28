package com.soywiz.kpspemu

object InstructionInterpreter : InstructionEvaluator<CpuState>(), InstructionDecoder {
	override fun addu(s: CpuState) = s { RD = RS + RT }
	override fun subu(s: CpuState) = s { RD = RS - RT }
	override fun addiu(s: CpuState) = s { RT = RS + U_IMM16 }
	override fun ori(s: CpuState) = s { RT = RS or IMM16 }
	//override fun syscall(s: CpuState) = s { TODO(); Unit }

}

