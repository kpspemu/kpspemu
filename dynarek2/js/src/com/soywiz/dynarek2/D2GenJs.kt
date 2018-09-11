package com.soywiz.dynarek2

import com.soywiz.dynarek2.target.js.*

actual fun D2Func.generate(context: D2Context, name: String?, debug: Boolean): D2Result {
    val body = generateJsBody(context, name, debug)

    //if (debug) {
    //    println("FUNCTION($name):")
    //    println(rbody.toString())
    //    println(generator.referencedFunctions)
    //}
    val funcGen = eval(body.body)
    val obj = js("({})")
    for ((k, v) in body.referencedFunctions) obj[k] = v
    val func = funcGen(obj)
    return D2Result(name, debug, body, func, free = { })
}

actual fun D2Context.registerDefaultFunctions() {
}

