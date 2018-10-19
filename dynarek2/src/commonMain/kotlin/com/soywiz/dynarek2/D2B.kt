package com.soywiz.dynarek2

import com.soywiz.dynarek2.*
import kotlin.reflect.*

inline operator fun KFunction<Boolean>.invoke(vararg args: D2Expr<*>): D2ExprB = D2Expr.Invoke(D2BOOL, this, *args)
