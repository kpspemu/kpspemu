package com.soywiz.dynarek2.target.js

import com.soywiz.dynarek2.*
import com.soywiz.korio.lang.*
import com.soywiz.korio.util.*

class JsGenerator(val context: D2Context, val name: String?, val debug: Boolean) {
    val ARG_NAMES = arrayOf("regs", "mem", "temp", "external")

    val referencedFunctions = LinkedHashMap<String, Any>()

    fun generateBody(f: D2Func): Indenter {
        return Indenter {
            generateStm(f.body)
        }
    }

    fun Indenter.generateStm(s: D2Stm?): Unit {
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
            is D2Stm.Write -> {
                line("${s.ref.access} = ${s.value.generateExpr()};")
            }
            else -> {
                TODO("${s::class.portableSimpleName} : $s")
            }
        }
        Unit
    }

    fun D2Expr<*>.generateExpr(): String = when (this) {
        is D2Expr.ILit -> "(${this.lit})"
        is D2Expr.IBinOp -> {
            val ls = "(" + this.l.generateExpr() + ")"
            val rs = "(" + this.r.generateExpr() + ")"
            val os = op.symbol
            when (this.op) {
                D2BinOp.MUL -> "Math.imul($ls, $rs)"
                else -> "(($ls $os $rs)|0)"
            }
        }
        is D2Expr.IComOp -> "(((${this.l.generateExpr()}) ${op.symbol} (${this.r.generateExpr()}))|0)"
        is D2Expr.InvokeI -> {
            val fname = "func_" + this.func.name
            referencedFunctions[fname] = this.func
            val argsStr = this.args.joinToString(", ") { it.generateExpr() }
            "$fname($argsStr)"
        }
        is D2Expr.Ref -> access
        else -> TODO("${this::class.portableSimpleName} : $this")
    }

    val D2Expr.Ref.access get() = "$accessBase[${offset.generateExpr()}]"
    val D2Expr.Ref.accessBase get() = "${memSlot.accessName}.${size.accessName}"

    val D2MemSlot.accessName: String get() = ARG_NAMES[index]

    val D2Size.accessName: String
        get() = when (this) {
            D2Size.BYTE -> "s8"
            D2Size.SHORT -> "s16"
            D2Size.INT -> "s32"
        }
}