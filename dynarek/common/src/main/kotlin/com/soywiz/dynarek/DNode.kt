package com.soywiz.dynarek

import kotlin.reflect.KClass
import kotlin.reflect.KMutableProperty1

interface DNode

interface DType<T : Any> : DNode {
	val clazz: KClass<T>
}

data class DClass<T : Any>(override val clazz: KClass<T>) : DType<T>
data class DPrimType<T : Any>(override val clazz: KClass<T>, val id: Int) : DType<T>

val DVOID = DPrimType<Unit>(Unit::class, 0)
val DINT = DPrimType<Int>(Int::class, 1)
val DFLOAT = DPrimType<Float>(Float::class, 2)
val DBOOL = DPrimType<Boolean>(Boolean::class, 3)

interface DExpr<T> : DNode
data class DLiteral<T>(val value: T) : DExpr<T>
data class DArg<T : Any>(val clazz: KClass<T>, val index: Int) : DExpr<T>
data class DBinopInt(val left: DExpr<Int>, val op: String, val right: DExpr<Int>) : DExpr<Int>

interface DRef<T> : DNode
data class DFieldAccess<T : Any, TR>(val clazz: KClass<T>, val obj: DExpr<T>, val prop: KMutableProperty1<T, TR>) : DExpr<TR>, DRef<TR>

interface DStm : DNode
data class DStms(val stms: List<DStm>) : DStm
data class DReturnExpr<T>(val expr: DExpr<T>) : DStm
data class DReturnVoid(val dummy: Boolean) : DStm
data class DAssign<T>(val left: DRef<T>, val value: DExpr<T>) : DStm

data class DIfElse(val cond: DExpr<Boolean>, val strue: DStm, var sfalse: DStm? = null) : DStm

