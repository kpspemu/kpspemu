package com.soywiz.dynarek2

import com.soywiz.dynarek2.target.js.*
import com.soywiz.korio.*
import com.soywiz.korio.util.*

actual fun D2Func.generate(context: D2Context, name: String?, debug: Boolean): D2Result {
    val generator = JsGenerator(context, name, debug)
    val body = generator.generateBody(this)
    val rname = name ?: "generated"
    val rbody = Indenter {
        line("(function $rname(funcs) {")
        indent {
            // functions
            for (k in generator.referencedFunctions.keys) {
                line("var $k = funcs.$k;")
            }
            line("return function $rname(regs, mem, temp, ext) {")
            indent {
                line(body)
            }
            line("};")
        }
        line("})")
    }

    //if (debug) {
    //    println("FUNCTION($name):")
    //    println(rbody.toString())
    //    println(generator.referencedFunctions)
    //}
    val funcGen = eval(rbody.toString())
    val func = funcGen(jsObject(*(generator.referencedFunctions.map { it.key to it.value }.toTypedArray())))
    return D2Result(body, func) { }
}

actual fun D2Context.registerDefaultFunctions() {
}

