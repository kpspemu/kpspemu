package com.soywiz.kpspemu

import org.junit.Test
import kotlin.test.Ignore

class InstructionTableTest {
	@Test
	@Ignore
	fun name() {
		val switch = DecodingTable.createSwitch(InstructionTable.instructions)
		println(switch)
	}

	@Test
	@Ignore
	fun name2() {
		DecodingTable.dumpImpl(InstructionTable.instructions)
	}

	@Test
	@Ignore
	fun name3() {
		DecodingTable.dumpInstructionList(InstructionTable.instructions)
	}
}