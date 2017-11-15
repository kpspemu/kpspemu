package com.soywiz.kpspemu.util

import com.soywiz.korio.stream.MemorySyncStream
import mytest.assertEquals
import org.junit.Test

class StructTest {
	data class Header(
		var magic: Int = 0,
		var size: Int = 0
	) {
		companion object : Struct<Header>({ Header() },
			Header::magic AS INT32,
			Header::size AS INT32
		)
	}

	data class Demo(
		var header: Header = Header(),
		var size: Int = 0
	) {
		companion object : Struct<Demo>({ Demo() },
			Demo::header AS Header,
			Demo::size AS INT32
		)
	}

	@Test
	fun name() {
		val demo = Demo(Header(1, 2), 3)
		val mem = MemorySyncStream()
		mem.write(Demo, demo)
		mem.position = 0
		val demo2 = mem.read(Demo)
		assertEquals(demo, demo2)
	}
}