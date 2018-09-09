package com.soywiz.dynarek2

expect fun D2Context.registerDefaultFunctions(): Unit

class D2FuncName(val name: String)

class D2Context {
    val funcs = LinkedHashMap<D2FuncName, Long>()

    init {
        registerDefaultFunctions()
    }

    fun getFunc(name: D2FuncName): Long = funcs[name] ?: error("Can't find function $name")

    fun registerFunc(name: D2FuncName, address: Long) {
        funcs[name] = address
    }
}

expect fun D2Func.generate(context: D2Context, name: String? = null, debug: Boolean = false): D2Result
