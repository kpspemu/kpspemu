package com.soywiz.dynarek

import kotlin.reflect.KClass
import kotlin.reflect.KMutableProperty1

class StmBuilder<TRet : Any, T0 : Any, T1 : Any>(val ret: KClass<TRet>, val t0: KClass<T0>, val t1: KClass<T1>) {
	inner class ElseBuilder(val ifElse: DIfElse) {
		infix fun ELSE(block: StmBuilder<TRet, T0, T1>.() -> Unit) {
			val b = createBuilder()
			block(b)
			ifElse.sfalse = b.build()
		}
	}

	val stms = ArrayList<DStm>()

	fun createBuilder() = StmBuilder<TRet, T0, T1>(ret, t0, t1)

	val <T> T.lit: DLiteral<T> get() = DLiteral(this)

	val p0 get() = DArg<T0>(t0, 0)
	val p1 get() = DArg<T1>(t1, 1)

	fun <T : Any> arg(clazz: KClass<T>, index: Int) = DArg<T>(clazz, index)

	operator fun DExpr<Int>.plus(that: DExpr<Int>) = DBinopInt(this, "+", that)
	operator fun DExpr<Int>.minus(that: DExpr<Int>) = DBinopInt(this, "-", that)
	operator fun DExpr<Int>.times(that: DExpr<Int>) = DBinopInt(this, "*", that)

	operator fun Int.plus(that: DExpr<Int>) = DBinopInt(this.lit, "+", that)
	operator fun Int.minus(that: DExpr<Int>) = DBinopInt(this.lit, "-", that)
	operator fun Int.times(that: DExpr<Int>) = DBinopInt(this.lit, "*", that)

	operator fun DExpr<Int>.plus(that: Int) = DBinopInt(this, "+", that.lit)
	operator fun DExpr<Int>.minus(that: Int) = DBinopInt(this, "-", that.lit)
	operator fun DExpr<Int>.times(that: Int) = DBinopInt(this, "*", that.lit)

	inline operator fun <reified T : Any, TR> DExpr<T>.get(prop: KMutableProperty1<T, TR>): DFieldAccess<T, TR> = DFieldAccess(T::class, this, prop)

	fun RET(expr: DExpr<TRet>) = stms.add(DReturnExpr(expr))
	fun RET() = stms.add(DReturnVoid(true))
	fun <T> SET(ref: DRef<T>, value: DExpr<T>) = stms.add(DAssign(ref, value))

	fun IF(cond: Boolean, block: StmBuilder<TRet, T0, T1>.() -> Unit): ElseBuilder = IF(cond.lit, block)

	fun IF(cond: DExpr<Boolean>, block: StmBuilder<TRet, T0, T1>.() -> Unit): ElseBuilder {
		val trueBuilder = createBuilder()
		block(trueBuilder)
		val ifElse = DIfElse(cond, trueBuilder.build())
		stms.add(ifElse)
		return ElseBuilder(ifElse)
	}

	fun build(): DStm = DStms(stms.toList())
}
