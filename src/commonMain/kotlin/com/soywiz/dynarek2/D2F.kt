package com.soywiz.dynarek2

import kotlin.reflect.*

inline operator fun KFunction<Float>.invoke(vararg args: D2Expr<*>): D2ExprF = D2Expr.Invoke(D2FLOAT, this, *args)

val Float.lit get() = D2Expr.FLit(this)

fun BINOP(l: D2ExprF, op: D2FBinOp, r: D2ExprF) = D2Expr.FBinOp(l, op, r)
//fun BINOP(l: D2ExprF, op: D2BinOp, r: D2ExprF) = D2Expr.FBinOp(l, op, r)

operator fun D2ExprF.plus(other: D2ExprF) = BINOP(this, D2FBinOp.ADD, other)
operator fun D2ExprF.minus(other: D2ExprF) = BINOP(this, D2FBinOp.SUB, other)
operator fun D2ExprF.times(other: D2ExprF) = BINOP(this, D2FBinOp.MUL, other)
operator fun D2ExprF.div(other: D2ExprF) = BINOP(this, D2FBinOp.DIV, other)
//operator fun D2ExprF.rem(other: D2ExprF) = BINOP(this, D2FBinOp.REM, other)
