package com.soywiz.dynarek2.target.jvm

import com.soywiz.dynarek2.*
import org.objectweb.asm.*
import kotlin.reflect.*

class JvmGenerator {
    val objectRef = java.lang.Object::class.asmRef
    val d2MemoryRef = D2Memory::class.asmRef

    fun generate(func: D2Func): D2Result {
        val cw = ClassWriter(0)
        val className = "Dynarek2Generated"

        cw.visit(
            Opcodes.V1_5,
            Opcodes.ACC_PUBLIC or Opcodes.ACC_FINAL,
            className,
            null,
            java.lang.Object::class.internalName,
            arrayOf(D2KFuncInt::class.java.internalName)
        )
        cw.apply {
            // constructor
            visitMethod(Opcodes.ACC_PUBLIC, "<init>", "()V", null, null).apply {
                visitMaxs(2, 1)
                visitVarInsn(Opcodes.ALOAD, 0) // push `this` to the operand stack
                visitMethodInsn(
                    Opcodes.INVOKESPECIAL,
                    java.lang.Object::class.internalName,
                    "<init>",
                    "()V",
                    false
                ) // call the constructor of super class
                visitInsn(Opcodes.RETURN)
                visitEnd()
            }
            // invoke method
            visitMethod(
                Opcodes.ACC_PUBLIC,
                "invoke",
                "($d2MemoryRef$d2MemoryRef$d2MemoryRef$objectRef)I",
                null,
                null
            ).apply {
                visitMaxs(32, 32)

                generate(func.body)

                // DUMMY to make valid labels
                pushInt(0)
                visitIReturn()

                visitEnd()
            }
        }

        cw.visitEnd()
        val classBytes = cw.toByteArray()

        try {
            val gclazz =
                createDynamicClass<D2KFuncInt>(
                    ClassLoader.getSystemClassLoader(),
                    className.replace('/', '.'),
                    classBytes
                )
            val instance = gclazz.declaredConstructors.first().newInstance() as D2KFuncInt

            return D2Result(classBytes, instance::invoke, { })
        } catch (e: VerifyError) {
            throw D2InvalidCodeGen("JvmGenerator", classBytes, e)
        }
    }

    fun MethodVisitor.generate(s: D2Stm): Unit = when (s) {
        is D2Stm.Stms -> for (c in s.children) generate(c)
        is D2Stm.Expr -> {
            generate(s.expr)
            visitPop()
        }
        is D2Stm.Return -> {
            generate(s.expr)
            visitIReturn()
        }
        is D2Stm.Write -> {
            val ref = s.ref
            visitIntInsn(Opcodes.ALOAD, ref.memSlot.index + 1) // 0 is used for THIS
            generate(ref.offset)
            generate(s.value)
            val methodName = when (ref.size) {
                D2Size.BYTE -> JvmMemTools::set8.name
                D2Size.SHORT -> JvmMemTools::set16.name
                D2Size.INT -> JvmMemTools::set32.name
            }
            visitMethodInsn(
                Opcodes.INVOKESTATIC,
                JvmMemTools::class.internalName,
                methodName,
                "(${d2MemoryRef}II)V",
                false
            )

        }
        is D2Stm.If -> {
            val cond = s.cond
            val strue = s.strue
            val sfalse = s.sfalse

            if (sfalse == null) {
                // IF
                val endLabel = Label()
                generateJumpFalse(cond, endLabel)
                generate(strue)
                visitLabel(endLabel)
            } else {
                // IF+ELSE
                val elseLabel = Label()
                val endLabel = Label()
                generateJumpFalse(cond, elseLabel)
                generate(strue)
                generateJumpAlways(endLabel)
                visitLabel(elseLabel)
                generate(sfalse)
                visitLabel(endLabel)
            }
        }
        is D2Stm.While -> {
            val startLabel = Label()
            val endLabel = Label()
            visitLabel(startLabel)
            generateJumpFalse(s.cond, endLabel)
            generate(s.body)
            generateJumpAlways(startLabel)
            visitLabel(endLabel)
        }
        is D2Stm.Print -> {
            generate(s.expr)
            visitMethodInsn(
                Opcodes.INVOKESTATIC,
                JvmMemTools::class.internalName,
                JvmMemTools::printi.name,
                "(I)V",
                false
            )
        }
        else -> TODO("$s")
    }

    fun MethodVisitor.generateJump(e: D2Expr<*>, label: Label, isTrue: Boolean): Unit {
        generate(e)
        visitJumpInsn(if (isTrue) Opcodes.IFNE else Opcodes.IFEQ, label)
    }

    fun MethodVisitor.generateJumpFalse(e: D2Expr<*>, label: Label): Unit = generateJump(e, label, false)
    fun MethodVisitor.generateJumpTrue(e: D2Expr<*>, label: Label): Unit = generateJump(e, label, true)
    fun MethodVisitor.generateJumpAlways(label: Label): Unit = visitJumpInsn(Opcodes.GOTO, label)

    fun MethodVisitor.generate(e: D2Expr<*>): Unit = when (e) {
        is D2Expr.ILit -> pushInt(e.lit)
        is D2Expr.IBinop -> {
            generate(e.l)
            generate(e.r)
            when (e.op) {
                D2Binop.ADD -> visitInsn(Opcodes.IADD)
                D2Binop.SUB -> visitInsn(Opcodes.ISUB)
                D2Binop.MUL -> visitInsn(Opcodes.IMUL)
                D2Binop.DIV -> visitInsn(Opcodes.IDIV)
                D2Binop.REM -> visitInsn(Opcodes.IREM)
            }
        }
        is D2Expr.IComop -> {
            generate(e.l)
            generate(e.r)
            val opcode = when (e.op) {
                D2Compop.EQ -> Opcodes.IF_ICMPEQ
                D2Compop.NE -> Opcodes.IF_ICMPNE
                D2Compop.LT -> Opcodes.IF_ICMPLT
                D2Compop.LE -> Opcodes.IF_ICMPLE
                D2Compop.GT -> Opcodes.IF_ICMPGT
                D2Compop.GE -> Opcodes.IF_ICMPGE
                else -> error("Invalid")
            }
            val label1 = Label()
            val label2 = Label()
            visitJumpInsn(opcode, label1)
            pushBool(false)
            visitJumpInsn(Opcodes.GOTO, label2)
            visitLabel(label1)
            pushBool(true)
            visitLabel(label2)
        }
        is D2Expr.IUnop -> {
            generate(e.l)
            when (e.op) {
                D2Unop.NEG -> visitInsn(Opcodes.INEG)
                D2Unop.INV -> TODO()
            }
        }
        is D2Expr.Ref -> {
            visitIntInsn(Opcodes.ALOAD, e.memSlot.index + 1) // 0 is used for THIS
            generate(e.offset)
            val methodName = when (e.size) {
                D2Size.BYTE -> JvmMemTools::get8.name
                D2Size.SHORT -> JvmMemTools::get16.name
                D2Size.INT -> JvmMemTools::get32.name
            }
            visitMethodInsn(
                Opcodes.INVOKESTATIC,
                JvmMemTools::class.internalName,
                methodName,
                "(${d2MemoryRef}I)I",
                false
            )
        }
        else -> TODO("$e")
    }

    fun <T> createDynamicClass(parent: ClassLoader, clazzName: String, b: ByteArray): Class<T> =
        OwnClassLoader(parent).defineClass(clazzName, b) as Class<T>

    private class OwnClassLoader(parent: ClassLoader) : ClassLoader(parent) {
        fun defineClass(name: String, b: ByteArray): Class<*> = defineClass(name, b, 0, b.size)
    }
}

object JvmMemTools {
    @JvmStatic
    fun get8(m: D2Memory, index: Int): Int = m.get8(index)

    @JvmStatic
    fun get16(m: D2Memory, index: Int): Int = m.get16(index)

    @JvmStatic
    fun get32(m: D2Memory, index: Int): Int = m.get32(index)

    @JvmStatic
    fun set8(m: D2Memory, index: Int, value: Int) = m.set8(index, value)

    @JvmStatic
    fun set16(m: D2Memory, index: Int, value: Int) = m.set16(index, value)

    @JvmStatic
    fun set32(m: D2Memory, index: Int, value: Int) = m.set32(index, value)

    @JvmStatic
    fun printi(i: Int): Unit = println(i)
}

val <T> Class<T>.internalName get() = Type.getInternalName(this)
val <T : Any> KClass<T>.internalName get() = this.java.internalName

val <T> Class<T>.asmRef get() = "L" + this.internalName + ";"
val <T : Any> KClass<T>.asmRef get() = this.java.asmRef

