package com.soywiz.dynarek

import org.junit.Test
import kotlin.test.assertEquals

class DrekTest {
	class State {
		var a = 0
		var b = 0
	}

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
}


