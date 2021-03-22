package com.soywiz.dynarek2

import kotlin.reflect.*

interface D2KFuncInt {
    operator fun invoke(regs: D2Memory?, mem: D2Memory?, temps: D2Memory?, external: Any?): Int
}

typealias D2KFunc = (regs: D2Memory?, mem: D2Memory?, temps: D2Memory?, external: Any?) -> Int

class D2InvalidCodeGen(val name: String, val data: ByteArray, val e: Throwable?) : Exception(e)

typealias InvalidCodeGenerated = D2InvalidCodeGen

data class D2Result(
    val name: String?,
    val debug: Boolean,
    val info: Any,
    val func: D2KFunc,
    val free: () -> Unit,
    val extra: Any? = null,
    val extraLong: Long = 0
)

expect class D2Runner() : D2BaseRunner

expect val isNativeWindows: Boolean

abstract class D2BaseRunner {
    private var regs: D2Memory? = null
    private var mem: D2Memory? = null
    private var temps: D2Memory? = null
    private var external: Any? = null
    private var result: D2Result? = null
    private var func: D2KFunc? = null

    open fun setParams(regs: D2Memory?, mem: D2Memory?, temps: D2Memory?, external: Any?) {
        this.regs = regs
        this.mem = mem
        this.temps = temps
        this.external = external
    }

    open fun setFunc(result: D2Result) {
        this.result = result
        this.func = result.func
    }

    open fun execute(): Int {
        val func = this.func
        if (func != null) {
            return func.invoke(regs, mem, temps, external)
        } else {
            return 0
        }
    }

    open fun close() {
    }
}

interface Dynarek2Generator {
    fun generate(func: D2Func): D2Result
}

class D2Func(val body: D2Stm.Stms)

enum class D2Size(val bytes: Int) {
    BYTE(1), SHORT(2), INT(4), FLOAT(4), LONG(8)
}

enum class D2MemSlot(val index: Int) { REGS(0), MEM(1), TEMPS(2) }

enum class D2IBinOp(val symbol: String) {
    ADD("+"), SUB("-"), MUL("*"), DIV("/"), REM("%"),
    OR("|"), AND("&"), XOR("^"),
    SHL("<<"), SHR(">>"), USHR(">>>"),
}

enum class D2FBinOp(val symbol: String) {
    ADD("+"), SUB("-"), MUL("*"), DIV("/")
    //, REM("%")
}

enum class D2CompOp(val symbol: String) {
    EQ("=="), NE("!="), LT("<"), LE("<="), GT(">"), GE(">=");

    val negated get() = when (this) {
        EQ -> NE
        NE -> EQ
        LT -> GE
        LE -> GT
        GT -> LE
        GE -> LT
    }
}

enum class D2UnOp {
    NEG, INV
}

open class D2TYPE<T>(val id: Int)

object D2INT : D2TYPE<Int>(0)
val D2BOOL = D2INT
object D2BYTE : D2TYPE<Int>(1)
object D2SHORT : D2TYPE<Int>(2)
object D2FLOAT : D2TYPE<Float>(3)
object D2LONG : D2TYPE<Long>(4)
object D2PTR : D2TYPE<Any>(5)

typealias D2ExprA = D2Expr<*>
typealias D2ExprB = D2Expr<Int> // Boolean
typealias D2ExprI = D2Expr<Int>
typealias D2ExprL = D2Expr<Long>
typealias D2ExprF = D2Expr<Float>

sealed class D2Stm {
    class Stms(val children: List<D2Stm>) : D2Stm()
    abstract class BExpr(val expr: D2ExprI) : D2Stm()
    class Expr(expr: D2ExprI) : BExpr(expr)
    class Return(expr: D2ExprI) : BExpr(expr)
    class If(val cond: D2ExprI, val strue: D2Stm, var sfalse: D2Stm? = null) : D2Stm()
    class While(val cond: D2ExprI, val body: D2Stm.Stms) : D2Stm()
    class Set<T>(val ref: D2Expr.Ref<T>, val value: D2Expr<T>) : D2Stm()
}

sealed class D2Expr<T>(val type: D2TYPE<T>) {
    abstract class B : D2Expr<Int>(D2INT)
    abstract class I : D2Expr<Int>(D2INT)
    abstract class L : D2Expr<Long>(D2LONG)
    abstract class F : D2Expr<Float>(D2FLOAT)

    class ILit(val lit: Int) : I()
    class LLit(val lit: Long) : L()
    class FLit(val lit: Float) : F()

    class LBinOp(val l: D2ExprL, val op: D2IBinOp, val r: D2ExprL) : L()
    class IBinOp(val l: D2ExprI, val op: D2IBinOp, val r: D2ExprI) : I()
    class FBinOp(val l: D2ExprF, val op: D2FBinOp, val r: D2ExprF) : F()

    class IComOp(val l: D2ExprI, val op: D2CompOp, val r: D2ExprI) : I()

    class IUnop(val l: D2ExprI, val op: D2UnOp) : I()
    class FUnop(val l: D2ExprF, val op: D2UnOp) : F()

    class Invoke<T>(type: D2TYPE<T>, val func: KFunction<*>, vararg val args: D2Expr<*>) : D2Expr<T>(type)
    class Ref<T>(val memSlot: D2MemSlot, type: D2TYPE<T>, val size: D2Size, val offset: D2ExprI) : D2Expr<T>(type)

    class External : D2Expr<Any>(D2PTR)

    companion object {
        fun RefI(memSlot: D2MemSlot, size: D2Size, offset: D2ExprI) = Ref(memSlot, D2INT, size, offset)
        fun RefF(memSlot: D2MemSlot, offset: D2ExprI) = Ref(memSlot, D2FLOAT, D2Size.FLOAT, offset)
        fun RefL(memSlot: D2MemSlot, offset: D2ExprI) = Ref(memSlot, D2LONG, D2Size.LONG, offset)
    }
}

val D2Expr<*>.isConstant get() = (this is D2Expr.ILit) || (this is D2Expr.LLit) || (this is D2Expr.FLit)

open class D2BuilderBase {
    // REGS

    //fun REGS32(offset: D2ExprI) = D2Expr.Read(D2MemSlot.REGS, D2Size.INT, offset)
    //fun SET_REGS32(offset: D2ExprI, value: D2ExprI) = run { stms += D2Stm.Write(D2MemSlot.REGS, D2Size.INT, offset, value) }

    //fun TEMP32(offset: D2ExprI) = D2Expr.Read(D2MemSlot.TEMPS, D2Size.INT, offset)
    //fun SET_TEMP32(offset: D2ExprI, value: D2ExprI) = run { stms += D2Stm.Write(D2MemSlot.TEMPS, D2Size.INT, offset, value) }

    val TRUE get() = 1.lit
    val FALSE get() = 0.lit

    //fun REGS32(offset: Int) = REGS32(offset.lit)
    //fun SET_REGS32(offset: Int, value: D2ExprI) = SET_REGS32(offset.lit, value)

    fun checkRegIndex(offset: Int): Int {
        if (offset < 0) error("Invalid register index $offset")
        return offset
    }

    fun REGS32(offset: Int) = D2Expr.RefI(D2MemSlot.REGS, D2Size.INT, checkRegIndex(offset).lit)
    fun REGS32(offset: D2ExprI) = D2Expr.RefI(D2MemSlot.REGS, D2Size.INT, offset)

    fun REGS64(offset: Int) = D2Expr.RefL(D2MemSlot.REGS, checkRegIndex(offset).lit)
    fun REGS64(offset: D2ExprI) = D2Expr.RefL(D2MemSlot.REGS, offset)

    fun REGF32(offset: Int) = D2Expr.RefF(D2MemSlot.REGS, checkRegIndex(offset).lit)
    fun REGF32(offset: D2ExprI) = D2Expr.RefF(D2MemSlot.REGS, offset)

    val EXTERNAL get() = D2Expr.External()

}

object D2ExprBuilder : D2BuilderBase() {
    inline operator fun <T> invoke(callback: D2ExprBuilder.() -> T): T = callback()
}

open class D2Builder : D2BuilderBase() {
    @PublishedApi
    internal val stms = arrayListOf<D2Stm>()

    // Stms

    fun IPRINT(e: D2ExprI) = STM(::iprint.invoke(e))

    fun STM(e: D2ExprI) = run { stms += D2Stm.Expr(e) }
    fun STM(st: List<D2Stm>) = run { this.stms += st }
    fun STM(st: D2Stm) = run { this.stms += st }
    fun RETURN(e: D2Expr<Int>) = run { stms += D2Stm.Return(e) }

    fun RET_VOID() = RETURN(0.lit)

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
    fun <T> SET(ref: D2Expr.Ref<T>, value: D2Expr<T>) = run { stms += D2Stm.Set(ref, value) }

    fun build() = D2Stm.Stms(stms.toList())

    inline operator fun <T> invoke(callback: D2Builder.() -> T): T = callback()

    fun FOR(counter: D2Expr.Ref<Int>, start: D2Expr.ILit, end: D2Expr.ILit, callback: D2Builder.() -> Unit) {
        SET(counter, start)
        WHILE(counter LT end) {
            callback()
            SET(counter, counter + 1.lit)
        }
    }

    companion object {
        operator fun invoke(callback: D2Builder.() -> Unit): D2Stm.Stms = D2Stm.Stms(D2Builder().apply(callback).stms)
    }
}

fun D2Func(generate: D2Builder.() -> Unit): D2Func = D2Func(D2Builder(generate))
