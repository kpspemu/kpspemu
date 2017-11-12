package com.soywiz.dynarek

fun <T> DExpr<T>.genJs(): String = when (this) {
	is DLiteral<*> -> "$value"
	is DArg<*> -> "p$index"
	is DBinopInt -> "((" + left.genJs() + " " + op + " " + right.genJs() + ")|0)" //
	else -> TODO("Unhandled.DExpr.genJs: $this")
}

fun DStm.genJs(w: StringBuilder): Unit = when (this) {
	is DReturnVoid -> run { w.append("return;"); Unit }
	is DReturnExpr<*> -> run { w.append("return " + expr.genJs() + ";"); Unit }
	is DAssign<*> -> {
		val l = left
		val r = value.genJs()
		when (l) {
			is DBindedProp<*, *> -> {
				val objs = l.obj.genJs()
				val propName = l.prop.name
				w.append("$objs.$propName = $r;")
			}
			else -> TODO("Unhandled.DStm.DAssign.genJs: $this")
		}
		Unit
	}
	is DStms -> for (stm in stms) stm.genJs(w)
	else -> TODO("Unhandled.DStm.genJs: $this")
}

@JsName("Function")
external class JsFunction(vararg args: dynamic)

private fun _generateDynarek(nargs: Int, func: DFunction): dynamic {
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

actual fun <TRet> DFunction0<TRet>.generateDynarek(): () -> TRet = _generateDynarek(0, this)
actual fun <TRet, T0> DFunction1<TRet, T0>.generateDynarek(): (T0) -> TRet = _generateDynarek(1, this)
actual fun <TRet, T0, T1> DFunction2<TRet, T0, T1>.generateDynarek(): (T0, T1) -> TRet = _generateDynarek(2, this)
