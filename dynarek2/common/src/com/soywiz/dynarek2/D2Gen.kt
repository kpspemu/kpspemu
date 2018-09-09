package com.soywiz.dynarek2

import com.soywiz.korio.*
import kotlin.math.*
import kotlin.reflect.*

expect fun D2Context.registerDefaultFunctions(): Unit

class D2FuncName(val name: String)

/*
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
*/

class D2Context {
    val funcs = LinkedHashMap<String, Long>()

    init {
        registerDefaultFunctions()
    }

    fun getFunc(name: KFunction<*>): Long = funcs[name.name] ?: error("Can't find function $name")

    fun registerFunc(name: KFunction<*>, address: Long) {
        funcs[name.name] = address
    }
}

fun iprint(v: Int): Int {
    //println("iprint:$v")
    return 0
}
fun isqrt(v: Int): Int {
    val res = sqrt(v.toDouble()).toInt()
    //println("isqrt($v)=$res")
    return res
}
fun isub(a: Int, b: Int): Int {
    val res = a - b
    //println("isub($a,$b)=$res")
    return res
}

expect fun D2Func.generate(context: D2Context, name: String? = null, debug: Boolean = false): D2Result
