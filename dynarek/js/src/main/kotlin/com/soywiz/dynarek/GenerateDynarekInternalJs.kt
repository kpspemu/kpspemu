package com.soywiz.dynarek

val <T> DExpr<T>.str: String
	get() = when (this) {
		is DLiteral<*> -> "$value"
		is DArg<*> -> "p$index"
		is DBinopInt -> "((${left.str} $op ${right.str})|0)"
		is DFieldAccess<*, *> -> "${obj.str}.${prop.name}"
		else -> TODO("Unhandled.DExpr.genJs: $this")
	}

fun DStm.genJs(w: StringBuilder): Unit = when (this) {
	is DReturnVoid -> run { w.append("return;"); Unit }
	is DReturnExpr<*> -> run { w.append("return ${expr.str};"); Unit }
	is DAssign<*> -> {
		val l = left
		val r = value.str
		when (l) {
			is DFieldAccess<*, *> -> {
				val objs = l.obj.str
				val propName = l.prop.name
				w.append("$objs.$propName = $r;")
			}
			else -> TODO("Unhandled.DStm.DAssign.genJs: $this")
		}
		Unit
	}
	is DStms -> for (stm in stms) stm.genJs(w)
	is DIfElse -> {
		w.append("if (${cond.str}) {")
		strue.genJs(w)
		w.append("}")
		if (sfalse != null) {
			w.append("else {")
			sfalse?.genJs(w)
			w.append("}")
		}
		Unit
	}
	else -> TODO("Unhandled.DStm.genJs: $this")
}

@JsName("Function")
external class JsFunction(vararg args: dynamic)

fun _generateDynarek(nargs: Int, func: DFunction): dynamic {
	val sb = StringBuilder()
	func.body.genJs(sb)

	// @TODO: Kotlin.JS: This produces syntax error!
	//val argNames = (0 until nargs).map { "p$it" }.toTypedArray()
	//return JsFunction(*argNames, sb.toString())

	return when (nargs) {
		0 -> JsFunction(sb.toString())
		1 -> JsFunction("p0", sb.toString())
		2 -> JsFunction("p0", "p1", sb.toString())
		else -> TODO("Unsupported args $nargs")
	}
}
