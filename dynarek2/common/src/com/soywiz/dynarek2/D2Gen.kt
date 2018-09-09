package com.soywiz.dynarek2

import com.soywiz.korio.*
import kotlin.math.*
import kotlin.reflect.*

expect fun D2Context.registerDefaultFunctions(): Unit

class D2FuncName(val name: String)

class D2Context {
    val funcs = LinkedHashMap<KFunction<*>, Long>()

    init {
        registerDefaultFunctions()
    }

    fun getFunc(name: KFunction<*>): Long = funcs[name] ?: error("Can't find function $name")

    fun registerFunc(name: KFunction<*>, address: Long) {
        funcs[name] = address
    }
}

fun isqrt(v: Int): Int = sqrt(v.toDouble()).toInt()
fun isub(a: Int, b: Int): Int = a - b

expect fun D2Func.generate(context: D2Context, name: String? = null, debug: Boolean = false): D2Result
