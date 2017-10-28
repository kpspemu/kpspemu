package com.soywiz.kpspemu.cpu.dis

import com.soywiz.kpspemu.mem.Memory

fun Memory.disasm(pc: Int): String = Disassembler.disasm(pc, this.lw(pc))
fun Memory.disasmMacro(pc: Int): String = Disassembler.disasmMacro(pc, this.lw(pc))
