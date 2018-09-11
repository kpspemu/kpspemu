package com.soywiz.dynarek2

import kotlin.reflect.*

inline operator fun KFunction<Int>.invoke(vararg args: D2Expr<*>): D2ExprI = D2Expr.Invoke(D2INT, this, *args)

val Int.lit get() = D2Expr.ILit(this)

operator fun D2ExprI.unaryMinus() = UNOP(this, D2UnOp.NEG)

fun BINOP(l: D2ExprI, op: D2IBinOp, r: D2ExprI) = D2Expr.IBinOp(l, op, r)
fun COMPOP(l: D2ExprI, op: D2CompOp, r: D2ExprI) = D2Expr.IComOp(l, op, r)
fun UNOP(l: D2ExprI, op: D2UnOp) = D2Expr.IUnop(l, op)

operator fun D2ExprI.plus(other: D2ExprI) = BINOP(this, D2IBinOp.ADD, other)
operator fun D2ExprI.minus(other: D2ExprI) = BINOP(this, D2IBinOp.SUB, other)
operator fun D2ExprI.times(other: D2ExprI) = BINOP(this, D2IBinOp.MUL, other)
operator fun D2ExprI.div(other: D2ExprI) = BINOP(this, D2IBinOp.DIV, other)
operator fun D2ExprI.rem(other: D2ExprI) = BINOP(this, D2IBinOp.REM, other)

infix fun D2ExprI.SHL(other: D2ExprI) = BINOP(this, D2IBinOp.SHL, other)
infix fun D2ExprI.SHR(other: D2ExprI) = BINOP(this, D2IBinOp.SHR, other)
infix fun D2ExprI.USHR(other: D2ExprI) = BINOP(this, D2IBinOp.USHR, other)

infix fun D2ExprI.AND(other: D2ExprI) = BINOP(this, D2IBinOp.AND, other)
infix fun D2ExprI.OR(other: D2ExprI) = BINOP(this, D2IBinOp.OR, other)
infix fun D2ExprI.XOR(other: D2ExprI) = BINOP(this, D2IBinOp.XOR, other)

infix fun D2ExprI.EQ(other: D2ExprI) = COMPOP(this, D2CompOp.EQ, other)
infix fun D2ExprI.NE(other: D2ExprI) = COMPOP(this, D2CompOp.NE, other)
infix fun D2ExprI.LT(other: D2ExprI) = COMPOP(this, D2CompOp.LT, other)
infix fun D2ExprI.LE(other: D2ExprI) = COMPOP(this, D2CompOp.LE, other)
infix fun D2ExprI.GT(other: D2ExprI) = COMPOP(this, D2CompOp.GT, other)
infix fun D2ExprI.GE(other: D2ExprI) = COMPOP(this, D2CompOp.GE, other)

private fun Int.mask(): Int = (1 shl this) - 1

fun D2ExprI.EXTRACT(offset: Int, size: Int): D2ExprI = (this SHR offset.lit) AND size.mask().lit
fun D2ExprI.INSERT(offset: Int, size: Int, value: D2ExprI): D2ExprI =
    (this AND (size.mask() shl offset).lit) OR ((value AND size.mask().lit) SHL offset.lit)

fun INV(v: D2ExprI) = v XOR (-1).lit
