package com.soywiz.dynarek2

interface D2KFuncInt {
    operator fun invoke(regs: D2Memory?, mem: D2Memory?, temps: D2Memory?, external: Any?): Int
}

typealias D2KFunc = (regs: D2Memory?, mem: D2Memory?, temps: D2Memory?, external: Any?) -> Int

class D2InvalidCodeGen(val name: String, val data: ByteArray, val e: Throwable?) : Exception(e)

data class D2Result(
    val info: Any,
    val func: D2KFunc,
    val free: () -> Unit
)

interface Dynarek2Generator {
    fun generate(func: D2Func): D2Result
}

class D2Func(val body: D2Stm.Stms)

enum class D2Size { BYTE, SHORT, INT }

enum class D2MemSlot(val index: Int) { REGS(0), MEM(1), TEMPS(2) }

sealed class D2Stm {
    class Stms(val children: List<D2Stm>) : D2Stm()
    class Expr(val expr: D2ExprI) : D2Stm()
    class Return(val expr: D2ExprI) : D2Stm()
    class If(val cond: D2ExprI, val strue: D2Stm, var sfalse: D2Stm? = null) : D2Stm()
    class While(val cond: D2ExprI, val body: D2Stm.Stms) : D2Stm()
    class Write(val ref: D2Expr.Ref, val value: D2ExprI) : D2Stm()
    class Print(val expr: D2ExprI) : D2Stm()
}

enum class D2Binop {
    ADD, SUB, MUL, DIV, REM
}

enum class D2Compop {
    EQ, NE, LT, LE, GT, GE
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
    abstract class I : D2Expr<Int>(D2INT)
    abstract class F : D2Expr<Float>(D2FLOAT)

    class ILit(val lit: Int) : I()
    class FLit(val lit: Float) : F()

    class IBinop(val l: D2ExprI, val op: D2Binop, val r: D2ExprI) : I()
    class FBinop(val l: D2ExprF, val op: D2Binop, val r: D2ExprF) : F()

    class IComop(val l: D2ExprI, val op: D2Compop, val r: D2ExprI) : I()

    class IUnop(val l: D2ExprI, val op: D2Unop) : I()
    class FUnop(val l: D2ExprF, val op: D2Unop) : F()

    class InvokeI(vararg args: D2Expr<*>) : I()
    class InvokeF(vararg args: D2Expr<*>) : F()

    class Ref(val memSlot: D2MemSlot, val size: D2Size, val offset: D2ExprI) : I()
}

open class D2Builder {
    @PublishedApi
    internal val stms = arrayListOf<D2Stm>()

    // Stms

    fun STM(e: D2ExprI) = run { stms += D2Stm.Expr(e) }
    fun RETURN(e: D2Expr<Int>) = run { stms += D2Stm.Return(e) }

    fun IF(cond: D2ExprI, body: D2Builder.() -> Unit): IfBuilder {
        val node = D2Stm.If(cond, D2Builder(body), null)
        stms += node
        return IfBuilder(node)
    }

    class IfBuilder(val node: D2Stm.If) {
        infix fun ELSE(callback: D2Builder.() -> Unit) {
            node.sfalse = D2Builder(callback)
        }
    }

    fun WHILE(cond: D2ExprI, body: D2Builder.() -> Unit): D2Stm.While = D2Stm.While(cond, D2Builder(body)).also { this.stms += it }
    fun SET(ref: D2Expr.Ref, value: D2ExprI) = run { stms += D2Stm.Write(ref, value) }
    fun PRINTI(ref: D2ExprI) = run { stms += D2Stm.Print(ref) }

    // Expressions

    fun COMPOP(l: D2ExprI, op: D2Compop, r: D2ExprI) = D2Expr.IComop(l, op, r)

    fun BINOP(l: D2ExprI, op: D2Binop, r: D2ExprI) = D2Expr.IBinop(l, op, r)
    fun BINOP(l: D2ExprF, op: D2Binop, r: D2ExprF) = D2Expr.FBinop(l, op, r)

    fun UNOP(l: D2ExprI, op: D2Unop) = D2Expr.IUnop(l, op)

    operator fun D2ExprI.plus(other: D2ExprI) = BINOP(this, D2Binop.ADD, other)
    operator fun D2ExprI.minus(other: D2ExprI) = BINOP(this, D2Binop.SUB, other)
    operator fun D2ExprI.times(other: D2ExprI) = BINOP(this, D2Binop.MUL, other)
    operator fun D2ExprI.div(other: D2ExprI) = BINOP(this, D2Binop.DIV, other)

    infix fun D2ExprI.EQ(other: D2ExprI) = COMPOP(this, D2Compop.EQ, other)
    infix fun D2ExprI.NE(other: D2ExprI) = COMPOP(this, D2Compop.NE, other)
    infix fun D2ExprI.LT(other: D2ExprI) = COMPOP(this, D2Compop.LT, other)
    infix fun D2ExprI.LE(other: D2ExprI) = COMPOP(this, D2Compop.LE, other)
    infix fun D2ExprI.GT(other: D2ExprI) = COMPOP(this, D2Compop.GT, other)
    infix fun D2ExprI.GE(other: D2ExprI) = COMPOP(this, D2Compop.GE, other)

    operator fun D2ExprI.unaryMinus() = UNOP(this, D2Unop.NEG)

    val Int.lit get() = D2Expr.ILit(this)
    val Float.lit get() = D2Expr.FLit(this)

    // REGS

    //fun REGS32(offset: D2ExprI) = D2Expr.Read(D2MemSlot.REGS, D2Size.INT, offset)
    //fun SET_REGS32(offset: D2ExprI, value: D2ExprI) = run { stms += D2Stm.Write(D2MemSlot.REGS, D2Size.INT, offset, value) }

    //fun TEMP32(offset: D2ExprI) = D2Expr.Read(D2MemSlot.TEMPS, D2Size.INT, offset)
    //fun SET_TEMP32(offset: D2ExprI, value: D2ExprI) = run { stms += D2Stm.Write(D2MemSlot.TEMPS, D2Size.INT, offset, value) }

    val TRUE get() = 1.lit
    val FALSE get() = 0.lit

    //fun REGS32(offset: Int) = REGS32(offset.lit)
    //fun SET_REGS32(offset: Int, value: D2ExprI) = SET_REGS32(offset.lit, value)

    fun REGS32(offset: Int) = D2Expr.Ref(D2MemSlot.REGS, D2Size.INT, offset.lit)
    fun REGS32(offset: D2ExprI) = D2Expr.Ref(D2MemSlot.REGS, D2Size.INT, offset)

    companion object {
        operator fun invoke(callback: D2Builder.() -> Unit): D2Stm.Stms = D2Stm.Stms(D2Builder().apply(callback).stms)
    }
}

fun D2Func(generate: D2Builder.() -> Unit): D2Func = D2Func(D2Builder(generate))
