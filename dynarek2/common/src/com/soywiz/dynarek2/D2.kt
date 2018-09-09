package com.soywiz.dynarek2

typealias D2KFunc = (regs: D2Memory?, mem: D2Memory?, temps: D2Memory?, external: Any?) -> Int

data class D2Result(
	val info: Any,
	val internalFunc: Any,
	val free: () -> Unit
)

interface Dynarek2Generator {
	fun generate(func: D2Func): D2Result
}

class D2Func(val body: D2Stm)

enum class D2Size { BYTE, SHORT, INT }

enum class D2MemSlot(val index: Int) { REGS(0), MEM(1), TEMPS(2) }

sealed class D2Stm {
	class Stms(val children: List<D2Stm>) : D2Stm()
	class Expr(val expr: D2ExprI) : D2Stm()
	class Return(val expr: D2ExprI) : D2Stm()
	class Write(val memSlot: D2MemSlot, val size: D2Size, val offset: D2ExprI, val value: D2ExprI) : D2Stm()
}

enum class D2Binop {
	ADD, SUB, MUL, DIV, REM
}

enum class D2Unop {
	NEG, INV
}

open class D2TYPE<T>(val id: Int)

object D2INT : D2TYPE<Int>(0)
object D2FLOAT : D2TYPE<Float>(1)

typealias D2ExprA = D2Expr<*>
typealias D2ExprI = D2Expr<Int>
typealias D2ExprF = D2Expr<Float>

sealed class D2Expr<T>(val type: D2TYPE<T>) {
	class ILit(val lit: Int) : D2Expr<Int>(D2INT)
	class FLit(val lit: Float) : D2Expr<Float>(D2FLOAT)

	class Binop<T>(val l: D2Expr<T>, val op: D2Binop, val r: D2Expr<T>) : D2Expr<T>(l.type)
	class Unop<T>(val l: D2Expr<T>, val op: D2Unop) : D2Expr<T>(l.type)

	class Invoke<T>(rettype: D2TYPE<T>, vararg args: D2Expr<*>): D2Expr<T>(rettype)

	class Read(val memSlot: D2MemSlot, val size: D2Size, offset: D2Expr<Int>): D2Expr<Int>(D2INT)
}

open class D2Builder {
	@PublishedApi
	internal val stms = arrayListOf<D2Stm>()

	fun STM(e: D2ExprI) = run { stms += D2Stm.Expr(e) }
	fun RETURN(e: D2Expr<Int>) = run { stms += D2Stm.Return(e) }

	fun <T> BINOP(l: D2Expr<T>, op: D2Binop, r: D2Expr<T>) = D2Expr.Binop(l, op, r)

	operator fun <T> D2Expr<T>.plus(other: D2Expr<T>) = D2Expr.Binop(this, D2Binop.ADD, other)
	operator fun <T> D2Expr<T>.minus(other: D2Expr<T>) = D2Expr.Binop(this, D2Binop.ADD, other)
	operator fun <T> D2Expr<T>.times(other: D2Expr<T>) = D2Expr.Binop(this, D2Binop.MUL, other)

	val Int.lit get() = D2Expr.ILit(this)
	val Float.lit get() = D2Expr.FLit(this)

	// REGS
	fun REGS32(offset: D2ExprI) = D2Expr.Read(D2MemSlot.REGS, D2Size.INT, offset)
	fun SET_REGS32(offset: D2ExprI, value: D2ExprI) = run { stms += D2Stm.Write(D2MemSlot.REGS, D2Size.INT, offset, value) }

	fun TEMP32(offset: D2ExprI) = D2Expr.Read(D2MemSlot.TEMPS, D2Size.INT, offset)
	fun SET_TEMP32(offset: D2ExprI, value: D2ExprI) = run { stms += D2Stm.Write(D2MemSlot.TEMPS, D2Size.INT, offset, value) }

	companion object {
		operator fun invoke(callback: D2Builder.() -> Unit): List<D2Stm> = D2Builder().apply(callback).stms
	}
}

fun D2Func(generate: D2Builder.() -> Unit): D2Func {
	return D2Func(D2Stm.Stms(D2Builder(generate)))
}

fun test() {
	D2Func {
		RETURN(1.lit + 2.lit)
	}
}