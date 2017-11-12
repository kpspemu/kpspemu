package com.soywiz.dynarek

import kotlin.reflect.KClass
import kotlin.reflect.KMutableProperty1

data class DFunction(val ret: DType<*>, val args: List<DType<*>>, val body: DStm)

interface DNode

interface DType<T> : DNode
data class DClass<T : Any>(val clazz: KClass<T>) : DType<T>
data class DPrimType<T>(val id: Int) : DType<T>

val DVOID = DPrimType<Unit>(0)
val DINT = DPrimType<Int>(1)
val DFLOAT = DPrimType<Float>(2)

interface DExpr<T> : DNode
data class DLiteral<T>(val value: T) : DExpr<T>
data class DArg<T>(val index: Int) : DExpr<T>
data class DBinop<T>(val left: DExpr<T>, val op: String, val right: DExpr<T>) : DExpr<T>

interface DRef<T> : DNode
data class DBindedProp<T, TR>(val obj: DExpr<T>, val prop: KMutableProperty1<T, TR>) : DExpr<TR>, DRef<TR>

interface DStm : DNode
data class DStms(val stms: List<DStm>) : DStm
data class DReturnExpr<T>(val expr: DExpr<T>) : DStm
data class DReturnVoid(val dummy: Boolean) : DStm
data class DAssign<T>(val left: DRef<T>, val value: DExpr<T>) : DStm


class StmBuilder<TRet, T0, T1, T2, T3> {
	val stms = ArrayList<DStm>()

	val <T> T.lit: DLiteral<T> get() = DLiteral(this)

	val p0 get() = DArg<T0>(0)
	val p1 get() = DArg<T1>(1)
	val p2 get() = DArg<T2>(2)
	val p3 get() = DArg<T3>(3)

	fun <T> arg(index: Int) = DArg<T>(index)

	operator fun DExpr<Int>.plus(that: DExpr<Int>) = DBinop(this, "+", that)
	operator fun DExpr<Int>.minus(that: DExpr<Int>) = DBinop(this, "-", that)
	operator fun DExpr<Int>.times(that: DExpr<Int>) = DBinop(this, "*", that)

	operator fun <T, TR> DExpr<T>.get(prop: KMutableProperty1<T, TR>): DBindedProp<T, TR> = DBindedProp(this, prop)

	fun RET(expr: DExpr<TRet>) = stms.add(DReturnExpr(expr))
	fun RET() = stms.add(DReturnVoid(true))
	fun <T> SET(ref: DRef<T>, value: DExpr<T>) = stms.add(DAssign(ref, value))

	fun build(): DStm = DStms(stms.toList())
}

//fun <TRet> function(ret: DType<TRet>, vararg args: DType<*>, block: StmBuilder<TRet, Unit, Unit, Unit, Unit>.() -> Unit): DFunction {
//	val builder = StmBuilder<TRet, Unit, Unit, Unit, Unit>()
//	block(builder)
//	return DFunction(ret, args.toList(), builder.build())
//}

fun <TRet> function(ret: DType<TRet>, block: StmBuilder<TRet, Unit, Unit, Unit, Unit>.() -> Unit): DFunction {
	val builder = StmBuilder<TRet, Unit, Unit, Unit, Unit>()
	block(builder)
	return DFunction(ret, listOf(), builder.build())
}

fun <TRet, T0> function(arg0: DType<T0>, ret: DType<TRet>, block: StmBuilder<TRet, T0, Unit, Unit, Unit>.() -> Unit): DFunction {
	val builder = StmBuilder<TRet, T0, Unit, Unit, Unit>()
	block(builder)
	return DFunction(ret, listOf(arg0), builder.build())
}

fun <TRet, T0, T1> function(arg0: DType<T0>, arg1: DType<T1>, ret: DType<TRet>, block: StmBuilder<TRet, T0, T1, Unit, Unit>.() -> Unit): DFunction {
	val builder = StmBuilder<TRet, T0, T1, Unit, Unit>()
	block(builder)
	return DFunction(ret, listOf(arg0, arg1), builder.build())
}

