package com.soywiz.dynarek2.target.js

import com.soywiz.dynarek2.*
import com.soywiz.dynarek2.tools.*

class JsGenerator(val context: D2Context, val name: String?, val debug: Boolean) {
    val REGS_NAME = "regs"
    val MEM_NAME = "mem"
    val TEMP_NAME = "temp"
    val EXTERNAL_NAME = "ext"

    val ARG_NAMES = arrayOf(REGS_NAME, MEM_NAME, TEMP_NAME, EXTERNAL_NAME)

    val referencedFunctions = LinkedHashMap<String, Any>()

    fun generateBody(f: D2Func): MiniIndenter {
        return MiniIndenter {
            generateStm(f.body)
        }
    }

    fun MiniIndenter.generateStm(s: D2Stm?): Unit {
        when (s) {
            null -> Unit
            is D2Stm.Stms -> for (child in s.children) generateStm(child)
            is D2Stm.Expr -> line("${s.expr.generateExpr()};")
            is D2Stm.Return -> line("return ${s.expr.generateExpr()};")
            is D2Stm.If -> {
                line("if (${s.cond.generateExpr()})") {
                    generateStm(s.strue)
                }
                if (s.sfalse != null) {
                    line("else") {
                        generateStm(s.sfalse)
                    }
                }
            }
            is D2Stm.While -> {
                line("while (${s.cond.generateExpr()})") {
                    generateStm(s.body)
                }
            }
            is D2Stm.Set<*> -> {
                line("${s.ref.access} = ${s.value.generateExpr()};")
            }
            else -> {
                TODO("$s")
            }
        }
        Unit
    }

    fun D2Expr<*>.generateExpr(): String = when (this) {
        is D2Expr.ILit -> "(${this.lit})"
        is D2Expr.FLit -> "Math.fround(${this.lit})"
        is D2Expr.IBinOp -> {
            val ls = "(" + this.l.generateExpr() + ")"
            val rs = "(" + this.r.generateExpr() + ")"
            val os = op.symbol
            when (this.op) {
                D2IBinOp.MUL -> "Math.imul($ls, $rs)"
                else -> "(($ls $os $rs)|0)"
            }
        }
        is D2Expr.FBinOp -> {
            val ls = "(" + this.l.generateExpr() + ")"
            val rs = "(" + this.r.generateExpr() + ")"
            val os = op.symbol
            "(($ls $os $rs))"
        }
        is D2Expr.IComOp -> "(((${this.l.generateExpr()}) ${op.symbol} (${this.r.generateExpr()}))|0)"
        is D2Expr.Invoke<*> -> {
            val fname = "func_" + this.func.name
            referencedFunctions[fname] = this.func
            val argsStr = this.args.joinToString(", ") { it.generateExpr() }
            "$fname($argsStr)"
        }
        is D2Expr.Ref -> access
        is D2Expr.External -> EXTERNAL_NAME
        else -> TODO("$this")
    }

    val D2Expr.Ref<*>.access get() = "$accessBase[${offset.generateExpr()}]"
    val D2Expr.Ref<*>.accessBase get() = "${memSlot.accessName}.${size.accessName}"

    val D2MemSlot.accessName: String get() = ARG_NAMES[index]

    val D2Size.accessName: String
        get() = when (this) {
            D2Size.BYTE -> "s8"
            D2Size.SHORT -> "s16"
            D2Size.INT -> "s32"
            D2Size.FLOAT -> "f32"
            D2Size.LONG -> "s64"
        }
}

class JsBody(
    val context: D2Context,
    val name: String?,
    val debug: Boolean,
    val generator: JsGenerator,
    val bodyIndenter: MiniIndenter
) {
    val referencedFunctions get() = generator.referencedFunctions
    val body by lazy { bodyIndenter.toString() }

    override fun toString(): String = body
}

fun D2Func.generateJsBody(context: D2Context, name: String? = null, debug: Boolean = false, strict: Boolean = true): JsBody {
    val generator = JsGenerator(context, name, debug)
    val body = generator.generateBody(this)
    val rname = name ?: "generated"
    val rbody = MiniIndenter {
        line("(function $rname(funcs) {")
        indent {
            if (strict) line("\"use strict\";")
            // functions
            for (k in generator.referencedFunctions.keys) {
                line("var $k = funcs.$k;")
            }
            line("return function $rname(regs, mem, temp, ext) {")
            indent {
                if (strict) line("\"use strict\";")
                line(body)
            }
            line("};")
        }
        line("})")
    }
    return JsBody(context, name, debug, generator, rbody)
}
