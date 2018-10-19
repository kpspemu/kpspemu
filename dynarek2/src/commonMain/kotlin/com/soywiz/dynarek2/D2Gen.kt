package com.soywiz.dynarek2

import kotlin.RuntimeException
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

class D2FuncInfo(val address: Long, val rettype: D2TYPE<*>, val args: Array<out D2TYPE<*>>)

class D2Context {
    val funcs = LinkedHashMap<String, D2FuncInfo>()

    init {
        registerDefaultFunctions()
    }

    fun getFunc(name: KFunction<*>): D2FuncInfo = funcs[name.name] ?: error("Can't find function $name")

    fun _registerFunc(name: KFunction<*>, address: Long, rettype: D2TYPE<*>, vararg args: D2TYPE<*>) {
        funcs[name.name] = D2FuncInfo(address, rettype, args)
    }

    val KClass<*>.d2type get() = when (this) {
        Float::class -> D2FLOAT
        Int::class -> D2INT
        else -> D2INT
    }
}

class MyClass {
    var demo = 7
    override fun toString(): String {
        return "MyClass(demo=$demo)"
    }

}

fun dyna_getDemo(cl: MyClass) = cl.demo

fun iprint(v: Int): Int {
    println("iprint:$v")
    return 0
}
fun isqrt(v: Int): Int {
    val res = sqrt(v.toDouble()).toInt()
    //println("isqrt($v)=$res")
    return res
}

class MyDemoException : RuntimeException()

fun demo_ithrow(): Int {
    throw MyDemoException()
}

fun isub(a: Int, b: Int): Int {
    val res = a - b
    //println("isub($a,$b)=$res")
    return res
}

fun fsub(a: Float, b: Float): Float {
    val res = a - b
    //println("isub($a,$b)=$res")
    return res
}

expect fun D2Func.generate(context: D2Context, name: String? = null, debug: Boolean = false): D2Result
