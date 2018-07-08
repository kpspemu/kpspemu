package com.soywiz.kpspemu.util

import com.soywiz.korio.stream.*
import com.soywiz.korio.util.*
import com.soywiz.kpspemu.*
import kotlin.test.*

class StructTest : BaseTest() {
    data class Header(
        var magic: Int = 0,
        var size: Int = 0
    ) {
        companion object : Struct<Header>(
            { Header() },
            Header::magic AS INT32,
            Header::size AS INT32
        )
    }

    enum class MyEnum(override val id: Int) : IdEnum {
        A(1), B(99), C(3);

        companion object : INT32_ENUM<MyEnum>(values())
    }

    data class Demo(
        var header: Header = Header(),
        var size: Int = 0,
        var enum: MyEnum = MyEnum.C
    ) {
        companion object : Struct<Demo>(
            { Demo() },
            Demo::header AS Header,
            Demo::size AS INT32,
            Demo::enum AS MyEnum
        )
    }

    @Test
    fun name() {
        val demo = Demo(Header(1, 2), 3, MyEnum.B)
        val mem = MemorySyncStream()
        mem.write(Demo, demo)
        mem.position = 0
        val demo2 = mem.read(Demo)
        assertEquals(demo, demo2)
    }
}