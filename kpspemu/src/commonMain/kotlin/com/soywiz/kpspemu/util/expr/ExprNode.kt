package com.soywiz.kpspemu.util.expr

import com.soywiz.kds.*
import com.soywiz.korio.error.*
import com.soywiz.korio.lang.*
import com.soywiz.korio.util.*
import kotlin.math.*
import kotlin.reflect.*
import com.soywiz.korio.error.invalidOp

interface ExprNode {
    @Suppress("UNCHECKED_CAST")
    open class EvalContext {
        open fun getProp(name: String): KMutableProperty0<*>? = TODO()
        open fun get(name: String): Any? = getProp(name)?.get()
        open fun set(name: String, value: Any?): Unit = run { (getProp(name) as? KMutableProperty0<Any?>?)?.set(value) }
    }

    fun eval(context: EvalContext): Any?

    data class VAR(val name: String) : ExprNode {
        override fun eval(context: EvalContext): Any? {
            return context.get(name)
        }
    }

    data class LIT(val value: Any?) : ExprNode {
        override fun eval(context: EvalContext): Any? = value
    }

    data class ARRAY_LIT(val items: List<ExprNode>) : ExprNode {
        override fun eval(context: EvalContext): Any? {
            return items.map { it.eval(context) }
        }
    }

    data class OBJECT_LIT(val items: List<Pair<ExprNode, ExprNode>>) : ExprNode {
        override fun eval(context: EvalContext): Any? {
            return items.map { it.first.eval(context) to it.second.eval(context) }.toMap()
        }
    }

    data class ACCESS(val expr: ExprNode, val name: ExprNode) : ExprNode {
        override fun eval(context: EvalContext): Any? {
            TODO()
            //val obj = expr.eval(context)
            //val key = name.eval(context)
            //return try {
            //	context.getAccess(obj, "$key")
            //} catch (t: Throwable) {
            //	null
            //}
        }
    }

    data class CALL(val method: ExprNode, val args: List<ExprNode>) : ExprNode {
        override fun eval(context: EvalContext): Any? {
            TODO()
            //al processedArgs = args.map { it.eval(context) }
            //hen (method) {
            //	is ExprNode.ACCESS -> {
            //		val obj = method.expr.eval(context)
            //		val methodName = method.name.eval(context)
            //		//println("" + obj + ":" + methodName)
            //		if (obj is Map<*, *>) {
            //			val k = obj[methodName]
            //			if (k is Template.DynamicInvokable) {
            //				return k.invoke(context, processedArgs)
            //			}
            //		}
            //		return obj.dynamicCallMethod(methodName, processedArgs.toTypedArray(), mapper = context.mapper)
            //	}
            //	is ExprNode.VAR -> {
            //		val func = context.config.functions[method.name]
            //		if (func != null) {
            //			return func.eval(processedArgs, context)
            //		}
            //	}
            //
            //eturn method.eval(context).dynamicCall(processedArgs.toTypedArray(), mapper = context.mapper)
        }
    }

    data class BINOP(val l: ExprNode, val r: ExprNode, val op: String) : ExprNode {
        override fun eval(context: EvalContext): Any? {
            val lr = l.eval(context)
            val rr = r.eval(context)
            return Dynamic2.binop(lr, rr, op)
        }
    }

    data class TERNARY(val cond: ExprNode, val etrue: ExprNode, val efalse: ExprNode) : ExprNode {
        override fun eval(context: EvalContext): Any? {
            return if (cond.eval(context).toDynamicBool()) {
                etrue.eval(context)
            } else {
                efalse.eval(context)
            }
        }
    }

    data class UNOP(val r: ExprNode, val op: String) : ExprNode {
        override fun eval(context: EvalContext): Any? {
            return when (op) {
                "", "+" -> r.eval(context)
                else -> Dynamic2.unop(r.eval(context), op)
            }
        }
    }

    companion object {
        fun parse(str: String): ExprNode {
            val tokens = ExprNode.Token.Companion.tokenize(str)
            return ExprNode.parseFullExpr(tokens)
        }

        fun parseId(r: ListReader<Token>): String {
            return r.read().text
        }

        fun expect(r: ListReader<Token>, vararg tokens: String) {
            val token = r.read()
            if (token.text !in tokens) invalidOp("Expected ${tokens.joinToString(", ")} but found $token")
        }

        fun parseFullExpr(r: ListReader<Token>): ExprNode {
            val result = ExprNode.parseExpr(r)
            if (r.hasMore && r.peek() !is ExprNode.Token.TEnd) {
                invalidOp("Expected expression at " + r.peek() + " :: " + r.list.map { it.text }.joinToString(""))
            }
            return result
        }

        private val BINOPS_PRIORITIES_LIST = listOf(
            listOf("*", "/", "%"),
            listOf("+", "-", "~"),
            listOf("==", "!=", "<", ">", "<=", ">=", "<=>"),
            listOf("&&"),
            listOf("||"),
            listOf("in"),
            listOf(".."),
            listOf("?:")
        )

        private val BINOPS = BINOPS_PRIORITIES_LIST.withIndex()
            .flatMap { (index, ops) -> ops.map { it to index } }
            .toMap()

        fun binopPr(str: String) = BINOPS[str] ?: 0

        fun parseBinExpr(r: ListReader<Token>): ExprNode {
            var result = parseFinal(r)
            while (r.hasMore) {
                //if (r.peek() !is ExprNode.Token.TOperator || r.peek().text !in ExprNode.BINOPS) break
                if (r.peek().text !in ExprNode.BINOPS) break
                val operator = r.read().text
                val right = parseFinal(r)
                if (result is BINOP) {
                    val a = result.l
                    val lop = result.op
                    val b = result.r
                    val rop = operator
                    val c = right
                    val lopPr = binopPr(lop)
                    val ropPr = binopPr(rop)
                    if (lopPr > ropPr) {
                        result = BINOP(a, BINOP(b, c, rop), lop)
                        continue
                    }
                }
                result = BINOP(result, right, operator)
            }
            return result
        }

        fun parseTernaryExpr(r: ListReader<Token>): ExprNode {
            var left = this.parseBinExpr(r)
            if (r.peek().text == "?") {
                r.skip();
                val middle = parseExpr(r)
                r.expect(":")
                val right = parseExpr(r)
                left = TERNARY(left, middle, right);
            }
            return left;
        }

        fun parseExpr(r: ListReader<Token>): ExprNode = parseTernaryExpr(r)

        private fun parseFinal(r: ListReader<Token>): ExprNode {
            val tok = r.peek().text.toUpperCase()
            var construct: ExprNode = when (tok) {
                "!", "~", "-", "+", "NOT" -> {
                    val op = tok
                    r.skip()
                    UNOP(
                        parseFinal(r), when (op) {
                            "NOT" -> "!"
                            else -> op
                        }
                    )
                }
                "(" -> {
                    r.read()
                    val result = ExprNode.parseExpr(r)
                    if (r.read().text != ")") throw RuntimeException("Expected ')'")
                    UNOP(result, "")
                }
            // Array literal
                "[" -> {
                    r.read()
                    val items = arrayListOf<ExprNode>()
                    loop@ while (r.hasMore && r.peek().text != "]") {
                        items += ExprNode.parseExpr(r)
                        when (r.peek().text) {
                            "," -> r.read()
                            "]" -> continue@loop
                            else -> invalidOp("Expected , or ]")
                        }
                    }
                    r.expect("]")
                    ARRAY_LIT(items)
                }
            // Object literal
                "{" -> {
                    r.read()
                    val items = arrayListOf<Pair<ExprNode, ExprNode>>()
                    loop@ while (r.hasMore && r.peek().text != "}") {
                        val k = ExprNode.parseFinal(r)
                        r.expect(":")
                        val v = ExprNode.parseExpr(r)
                        items += k to v
                        when (r.peek().text) {
                            "," -> r.read()
                            "}" -> continue@loop
                            else -> invalidOp("Expected , or }")
                        }
                    }
                    r.expect("}")
                    OBJECT_LIT(items)
                }
                else -> {
                    // Number
                    if (r.peek() is ExprNode.Token.TNumber) {
                        val ntext = r.read().text
                        if (ntext.startsWith("0x")) {
                            LIT(ntext.substr(2).toIntOrNull(16) ?: 0)
                        } else if (ntext.contains('.')) {
                            LIT(ntext.toDoubleOrNull() ?: 0.0)
                        } else {
                            when (ntext.toLongOrNull()) {
                                ntext.toIntOrNull()?.toLong() -> LIT(ntext.toIntOrNull() ?: 0)
                                ntext.toLongOrNull() -> LIT(ntext.toLongOrNull() ?: 0L)
                                else -> LIT(ntext.toDoubleOrNull() ?: 0.0)
                            }
                        }
                    }
                    // String
                    else if (r.peek() is ExprNode.Token.TString) {
                        LIT((r.read() as Token.TString).processedValue)
                    }
                    // ID
                    else {
                        VAR(r.read().text)
                    }
                }
            }

            loop@ while (r.hasMore) {
                when (r.peek().text) {
                    "." -> {
                        r.read()
                        val id = r.read().text
                        construct = ACCESS(construct, LIT(id))
                        continue@loop
                    }
                    "[" -> {
                        r.read()
                        val expr = ExprNode.parseExpr(r)
                        construct = ACCESS(construct, expr)
                        val end = r.read()
                        if (end.text != "]") throw RuntimeException("Expected ']' but found $end")
                    }
                    "(" -> {
                        r.read()
                        val args = arrayListOf<ExprNode>()
                        callargsloop@ while (r.hasMore && r.peek().text != ")") {
                            args += ExprNode.parseExpr(r)
                            when (r.expectPeek(",", ")").text) {
                                "," -> r.read()
                                ")" -> break@callargsloop
                            }
                        }
                        r.expect(")")
                        construct = CALL(construct, args)
                    }
                    else -> break@loop
                }
            }
            return construct
        }
    }

    interface Token {
        val text: String

        data class TId(override val text: String) : ExprNode.Token
        data class TNumber(override val text: String) : ExprNode.Token
        data class TString(override val text: String, val processedValue: String) : ExprNode.Token
        data class TOperator(override val text: String) : ExprNode.Token
        data class TEnd(override val text: String = "") : ExprNode.Token

        companion object {
            private val OPERATORS = setOf(
                "(", ")",
                "[", "]",
                "{", "}",
                "&&", "||",
                "&", "|", "^",
                "==", "!=", "<", ">", "<=", ">=", "<=>",
                "?:",
                "..",
                "+", "-", "*", "/", "%", "**",
                "!", "~",
                ".", ",", ";", ":", "?",
                "="
            )

            fun tokenize(str: String): ListReader<Token> {
                val r = StrReader(str)
                val out = arrayListOf<ExprNode.Token>()
                fun emit(str: ExprNode.Token) {
                    out += str
                }
                while (r.hasMore) {
                    val start = r.pos
                    r.skipSpaces()
                    val id = r.readWhile(Char::isLetterDigitOrUnderscore)
                    if (id.isNotEmpty()) {
                        if (id[0].isDigit()) emit(ExprNode.Token.TNumber(id)) else emit(ExprNode.Token.TId(id))
                    }
                    r.skipSpaces()
                    if (r.peek(3) in ExprNode.Token.Companion.OPERATORS) emit(ExprNode.Token.TOperator(r.read(3)))
                    if (r.peek(2) in ExprNode.Token.Companion.OPERATORS) emit(ExprNode.Token.TOperator(r.read(2)))
                    if (r.peek(1) in ExprNode.Token.Companion.OPERATORS) emit(ExprNode.Token.TOperator(r.read(1)))
                    if (r.peek() == '\'' || r.peek() == '"') {
                        val strStart = r.read()
                        val strBody = r.readUntil(strStart) ?: ""
                        val strEnd = r.read()
                        emit(ExprNode.Token.TString(strStart + strBody + strEnd, strBody.unescape()))
                    }
                    val end = r.pos
                    if (end == start) invalidOp("Don't know how to handle '${r.peek()}'")
                }
                emit(ExprNode.Token.TEnd())
                return ListReader(out)
            }
        }
    }
}

fun ListReader<ExprNode.Token>.tryRead(vararg types: String): ExprNode.Token? {
    val token = this.peek()
    if (token.text in types) {
        this.read()
        return token
    } else {
        return null
    }
}

fun ListReader<ExprNode.Token>.expectPeek(vararg types: String): ExprNode.Token {
    val token = this.peek()
    if (token.text !in types) throw kotlin.RuntimeException("Expected ${types.joinToString(", ")} but found '${token.text}'")
    return token
}

fun ListReader<ExprNode.Token>.expect(vararg types: String): ExprNode.Token {
    val token = this.read()
    if (token.text !in types) throw kotlin.RuntimeException("Expected ${types.joinToString(", ")}")
    return token
}

fun ListReader<ExprNode.Token>.parseExpr() = ExprNode.parseExpr(this)
fun ListReader<ExprNode.Token>.parseId() = ExprNode.parseId(this)
fun ListReader<ExprNode.Token>.parseIdList(): List<String> {
    val ids = kotlin.collections.arrayListOf<String>()
    do {
        ids += parseId()
    } while (tryRead(",") != null)
    return ids
}

object Dynamic2 {

    fun binop(l: Any?, r: Any?, op: String): Any? = when (op) {
        "+" -> {
            when (l) {
                is String -> l.toString() + r.toString()
                is Iterable<*> -> toIterable(l) + toIterable(r)
                else -> toDouble(l) + toDouble(r)
            }
        }
        "-" -> toDouble(l) - toDouble(r)
        "*" -> toDouble(l) * toDouble(r)
        "/" -> toDouble(l) / toDouble(r)
        "%" -> toDouble(l) % toDouble(r)
        "**" -> toDouble(l).pow(toDouble(r))
        "&" -> toInt(l) and toInt(r)
        "or" -> toInt(l) or toInt(r)
        "^" -> toInt(l) xor toInt(r)
        "&&" -> toBool(l) && toBool(r)
        "||" -> toBool(l) || toBool(r)
        "==" -> {
            if (l is Number && r is Number) {
                l.toDouble() == r.toDouble()
            } else {
                l == r
            }
        }
        "!=" -> {
            if (l is Number && r is Number) {
                l.toDouble() != r.toDouble()
            } else {
                l != r
            }
        }
        "<" -> compare(l, r) < 0
        "<=" -> compare(l, r) <= 0
        ">" -> compare(l, r) > 0
        ">=" -> compare(l, r) >= 0
        "in" -> contains(r, l)
        "?:" -> if (toBool(l)) l else r
        else -> noImpl("Not implemented binary operator '$op'")
    }

    fun unop(r: Any?, op: String): Any? = when (op) {
        "+" -> r
        "-" -> -toDouble(r)
        "~" -> toInt(r).inv()
        "!" -> !toBool(r)
        else -> noImpl("Not implemented unary operator $op")
    }

    fun toBool(it: Any?): Boolean = when (it) {
        null -> false
        else -> toBoolOrNull(it) ?: true
    }

    fun toBoolOrNull(it: Any?): Boolean? = when (it) {
        null -> null
        is Boolean -> it
        is Number -> it.toDouble() != 0.0
        is String -> it.isNotEmpty() && it != "0" && it != "false"
        else -> null
    }

    fun toNumber(it: Any?): Number = when (it) {
        null -> 0.0
        is Number -> it
        else -> it.toString().toDoubleOrNull() ?: 0.0
    }

    fun toInt(it: Any?): Int = when (it) {
        is Int -> it
        is Long -> (it and 0xFFFFFFFFL).toInt()
        is Double -> toInt(it.toLong())
        else -> toNumber(it).toInt()
    }

    fun toLong(it: Any?): Long = toNumber(it).toLong()
    fun toDouble(it: Any?): Double = toNumber(it).toDouble()

    fun compare(l: Any?, r: Any?): Int {
        if (l is Number && r is Number) {
            return l.toDouble().compareTo(r.toDouble())
        }
        val lc = toComparable(l)
        val rc = toComparable(r)
        if (lc::class.isInstance(rc)) {
            return lc.compareTo(rc)
        } else {
            return -1
        }
    }

    @Suppress("UNCHECKED_CAST")
    fun toComparable(it: Any?): Comparable<Any?> = when (it) {
        null -> 0 as Comparable<Any?>
        is Comparable<*> -> it as Comparable<Any?>
        else -> it.toString() as Comparable<Any?>
    }

    fun toList(it: Any?): List<*> = toIterable(it).toList()

    fun toIterable(it: Any?): Iterable<*> = when (it) {
        null -> listOf<Any?>()
        is Iterable<*> -> it
        is CharSequence -> it.toList()
        is Map<*, *> -> it.toList()
        else -> listOf<Any?>()
    }

    fun contains(collection: Any?, element: Any?): Boolean = when (collection) {
        is Set<*> -> element in collection
        else -> element in toList(collection)
    }

    fun toString(value: Any?): String = when (value) {
        null -> ""
        is String -> value
        is Double -> {
            if (value == value.toInt().toDouble()) {
                value.toInt().toString()
            } else {
                value.toString()
            }
        }
        is Iterable<*> -> "[" + value.map { toString(it) }.joinToString(", ") + "]"
        is Map<*, *> -> "{" + value.map { toString(it.key).quote() + ": " + toString(it.value) }.joinToString(", ") + "}"
        else -> value.toString()
    }
}

internal fun Any?.toDynamicString() = Dynamic2.toString(this)
internal fun Any?.toDynamicBool() = Dynamic2.toBool(this)
internal fun Any?.toDynamicInt() = Dynamic2.toInt(this)
internal fun Any?.toDynamicList() = Dynamic2.toList(this)
//internal fun Any?.dynamicLength() = Dynamic2.length(this)
//internal fun Any?.dynamicGet(key: Any?, mapper: ObjectMapper2) = Dynamic2.accessAny(this, key, mapper)
//internal fun Any?.dynamicSet(key: Any?, value: Any?, mapper: ObjectMapper2) = Dynamic2.setAny(this, key, value, mapper)
//internal fun Any?.dynamicCall(vararg args: Any?, mapper: ObjectMapper2) = Dynamic2.callAny(this, args.toList(), mapper = mapper)
//internal fun Any?.dynamicCallMethod(methodName: Any?, vararg args: Any?, mapper: ObjectMapper2) = Dynamic2.callAny(this, methodName, args.toList(), mapper = mapper)
//internal fun Any?.dynamicCastTo(target: KClass<*>) = Dynamic2.dynamicCast(this, target)

