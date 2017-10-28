package com.soywiz.kpspemu

import com.soywiz.korio.util.extract

interface InstructionDecoder {
	val Int.rs: Int get() = TODO()
	val Int.rd: Int get() = TODO()
	val Int.rt: Int get() = TODO()

	operator fun <T> CpuState.invoke(callback: CpuState.() -> T) = this.run(callback)

	var CpuState.RD: Int; get() = this.GPR[this.I.rd]; set(value) = run { this.GPR[this.I.rd] = value }
	var CpuState.RS: Int; get() = this.GPR[this.I.rs]; set(value) = run { this.GPR[this.I.rs] = value }
	var CpuState.RT: Int; get() = this.GPR[this.I.rt]; set(value) = run { this.GPR[this.I.rt] = value }

	//val CpuState.IMM16: Int; get() = I.extract()
	//val CpuState.U_IMM16: Int get() = TODO()

	val CpuState.IMM16: Int; get() = TODO()
	val CpuState.U_IMM16: Int get() = TODO()
}
