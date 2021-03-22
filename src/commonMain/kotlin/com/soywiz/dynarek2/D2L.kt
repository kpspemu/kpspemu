package com.soywiz.dynarek2

import kotlin.reflect.*

inline operator fun KFunction<Long>.invoke(vararg args: D2Expr<*>): D2ExprL = D2Expr.Invoke(D2LONG, this, *args)

val Long.lit get() = D2Expr.LLit(this)

fun BINOP(l: D2ExprL, op: D2IBinOp, r: D2ExprL) = D2Expr.LBinOp(l, op, r)

operator fun D2ExprL.plus(other: D2ExprL) = BINOP(this, D2IBinOp.ADD, other)
operator fun D2ExprL.minus(other: D2ExprL) = BINOP(this, D2IBinOp.SUB, other)
operator fun D2ExprL.times(other: D2ExprL) = BINOP(this, D2IBinOp.MUL, other)
operator fun D2ExprL.div(other: D2ExprL) = BINOP(this, D2IBinOp.DIV, other)
operator fun D2ExprL.rem(other: D2ExprL) = BINOP(this, D2IBinOp.REM, other)
