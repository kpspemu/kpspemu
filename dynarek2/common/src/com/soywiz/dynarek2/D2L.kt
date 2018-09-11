package com.soywiz.dynarek2

import kotlin.reflect.*

inline operator fun KFunction<Long>.invoke(vararg args: D2Expr<*>): D2ExprL = D2Expr.Invoke(D2LONG, this, *args)

val Long.lit get() = D2Expr.LLit(this)
