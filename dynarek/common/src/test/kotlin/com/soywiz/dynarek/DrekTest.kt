package com.soywiz.dynarek

import org.junit.Test
import kotlin.test.assertEquals

class DrekTest {
	@Test
	fun name() {
		val function = function(DClass(State::class), DINT, DVOID) {
			SET(p0[State::a], p1 * 2.lit)
		}

		val state = State()
		val interpreter = DSlowInterpreter(listOf(state, 10))
		interpreter.interpret(function)
		assertEquals(20, state.a)
	}

	@Test
	fun name2() {
		val function = function(DClass(State::class), DINT, DVOID) {
			//IF(true) {
			SET(p0[State::a], p0[State::a] + 4 * p1)
			//} ELSE {
			//	SET(p0[State::b], 4 * p1)
			//}
		}
		val state = State(a = 7)
		val func = function.generateDynarek()
		val ret = func(state, 2)

		assertEquals(15, state.a)
	}
}

