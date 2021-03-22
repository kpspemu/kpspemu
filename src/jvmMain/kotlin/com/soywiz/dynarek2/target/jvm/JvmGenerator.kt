package com.soywiz.dynarek2.target.jvm

import com.soywiz.dynarek2.*
import org.objectweb.asm.*
import org.objectweb.asm.Type
import java.lang.reflect.*
import kotlin.reflect.*

class JvmGenerator {
    val objectRef = java.lang.Object::class.asmRef
    val d2MemoryRef = D2Memory::class.asmRef
    //val classLoader = this::class.javaClass.classLoader

    // 0: this
    // 1: regs
    // 2: mem
    // 3: temp
    // 4: external

    fun generate(func: D2Func, context: D2Context, name: String?, debug: Boolean): D2Result {
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

            return D2Result(name, debug, classBytes, instance::invoke, free = { })
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
        is D2Stm.Set<*> -> {
            val ref = s.ref
            visitIntInsn(Opcodes.ALOAD, ref.memSlot.jArgIndex)
            generate(ref.offset)
            generate(s.value)
            val methodName = when (ref.size) {
                D2Size.BYTE -> JvmMemTools::set8.name
                D2Size.SHORT -> JvmMemTools::set16.name
                D2Size.INT -> JvmMemTools::set32.name
                D2Size.FLOAT -> JvmMemTools::setF32.name
                D2Size.LONG -> JvmMemTools::set64.name
            }
            val primType = ref.size.jPrimType
            val signature = "(${d2MemoryRef}I$primType)V"
            //println("signature:$signature")
            visitMethodInsn(
                Opcodes.INVOKESTATIC,
                JvmMemTools::class.internalName,
                methodName,
                signature,
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
        else -> TODO("$s")
    }

    val D2Size.jPrimType get() = when (this) {
        D2Size.BYTE, D2Size.SHORT, D2Size.INT -> 'I'
        D2Size.FLOAT -> 'F'
        D2Size.LONG -> 'J'
    }

    fun MethodVisitor.generateJump(e: D2Expr<*>, label: Label, isTrue: Boolean): Unit {
        generate(e)
        visitJumpInsn(if (isTrue) Opcodes.IFNE else Opcodes.IFEQ, label)
    }

    fun MethodVisitor.generateJumpFalse(e: D2Expr<*>, label: Label): Unit = generateJump(e, label, false)
    fun MethodVisitor.generateJumpTrue(e: D2Expr<*>, label: Label): Unit = generateJump(e, label, true)
    fun MethodVisitor.generateJumpAlways(label: Label): Unit = visitJumpInsn(Opcodes.GOTO, label)

    val D2MemSlot.jArgIndex get() = this.index + 1 // 0 is used for THIS

    fun MethodVisitor.generate(e: D2Expr<*>): Unit {
        when (e) {
            is D2Expr.ILit -> pushInt(e.lit)
            is D2Expr.FLit -> pushFloat(e.lit)
            is D2Expr.LLit -> pushLong(e.lit)
            is D2Expr.IBinOp -> {
                generate(e.l)
                generate(e.r)
                when (e.op) {
                    D2IBinOp.ADD -> visitInsn(Opcodes.IADD)
                    D2IBinOp.SUB -> visitInsn(Opcodes.ISUB)
                    D2IBinOp.MUL -> visitInsn(Opcodes.IMUL)
                    D2IBinOp.DIV -> visitInsn(Opcodes.IDIV)
                    D2IBinOp.REM -> visitInsn(Opcodes.IREM)
                    D2IBinOp.SHL -> visitInsn(Opcodes.ISHL)
                    D2IBinOp.SHR -> visitInsn(Opcodes.ISHR)
                    D2IBinOp.USHR -> visitInsn(Opcodes.IUSHR)
                    D2IBinOp.OR -> visitInsn(Opcodes.IOR)
                    D2IBinOp.AND -> visitInsn(Opcodes.IAND)
                    D2IBinOp.XOR -> visitInsn(Opcodes.IXOR)
                }
            }
            is D2Expr.LBinOp -> {
                generate(e.l)
                generate(e.r)
                when (e.op) {
                    D2IBinOp.ADD -> visitInsn(Opcodes.LADD)
                    D2IBinOp.SUB -> visitInsn(Opcodes.LSUB)
                    D2IBinOp.MUL -> visitInsn(Opcodes.LMUL)
                    D2IBinOp.DIV -> visitInsn(Opcodes.LDIV)
                    D2IBinOp.REM -> visitInsn(Opcodes.LREM)
                    D2IBinOp.SHL -> visitInsn(Opcodes.LSHL)
                    D2IBinOp.SHR -> visitInsn(Opcodes.LSHR)
                    D2IBinOp.USHR -> visitInsn(Opcodes.LUSHR)
                    D2IBinOp.OR -> visitInsn(Opcodes.LOR)
                    D2IBinOp.AND -> visitInsn(Opcodes.LAND)
                    D2IBinOp.XOR -> visitInsn(Opcodes.LXOR)
                }
            }
            is D2Expr.FBinOp -> {
                generate(e.l)
                generate(e.r)
                when (e.op) {
                    D2FBinOp.ADD -> visitInsn(Opcodes.FADD)
                    D2FBinOp.SUB -> visitInsn(Opcodes.FSUB)
                    D2FBinOp.MUL -> visitInsn(Opcodes.FMUL)
                    D2FBinOp.DIV -> visitInsn(Opcodes.FDIV)
                    //D2FBinOp.REM -> visitInsn(Opcodes.FREM)
                }
            }
            is D2Expr.IComOp -> {
                generate(e.l)
                generate(e.r)
                val opcode = when (e.op) {
                    D2CompOp.EQ -> Opcodes.IF_ICMPEQ
                    D2CompOp.NE -> Opcodes.IF_ICMPNE
                    D2CompOp.LT -> Opcodes.IF_ICMPLT
                    D2CompOp.LE -> Opcodes.IF_ICMPLE
                    D2CompOp.GT -> Opcodes.IF_ICMPGT
                    D2CompOp.GE -> Opcodes.IF_ICMPGE
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
                    D2UnOp.NEG -> visitInsn(Opcodes.INEG)
                    D2UnOp.INV -> TODO()
                }
            }
            is D2Expr.Ref -> {
                visitIntInsn(Opcodes.ALOAD, e.memSlot.jArgIndex)
                generate(e.offset)
                val methodName = when (e.size) {
                    D2Size.BYTE -> JvmMemTools::get8.name
                    D2Size.SHORT -> JvmMemTools::get16.name
                    D2Size.INT -> JvmMemTools::get32.name
                    D2Size.LONG -> JvmMemTools::get64.name
                    D2Size.FLOAT -> JvmMemTools::getF32.name
                }
                val retType = e.size.jPrimType
                visitMethodInsn(
                    Opcodes.INVOKESTATIC,
                    JvmMemTools::class.internalName,
                    methodName,
                    "(${d2MemoryRef}I)$retType",
                    false
                )
            }
            is D2Expr.Invoke<*> -> {
                val func = e.func
                val name = func.name
                val owner = func.owner
                val smethod = func.method

                if (smethod.parameterCount != e.args.size) error("Arity mismatch generating method call")

                val signature = "(" + smethod.signature.substringAfter('(')
                for ((index, arg) in e.args.withIndex()) {
                    generate(arg)
                    val param = smethod.parameterTypes[index]
                    if (!param.isPrimitive) {
                        //println("PARAM: $param")
                        visitTypeInsn(Opcodes.CHECKCAST, param.internalName)
                    }
                }
                visitMethodInsn(Opcodes.INVOKESTATIC, owner.internalName, name, signature, false)
            }
            is D2Expr.External -> {
                visitVarInsn(Opcodes.ALOAD, 4)
            }
            else -> TODO("$e")
        }
        return
    }

    fun <T> createDynamicClass(parent: ClassLoader, clazzName: String, b: ByteArray): Class<T> =
        OwnClassLoader(parent).defineClass(clazzName, b) as Class<T>

    private class OwnClassLoader(parent: ClassLoader) : ClassLoader(parent) {
        fun defineClass(name: String, b: ByteArray): Class<*> = defineClass(name, b, 0, b.size)
    }
}

val KFunction<*>.owner: Class<*> get() = (javaClass.methods.first { it.name == "getOwner" }.apply { isAccessible = true }.invoke(this) as kotlin.jvm.internal.PackageReference).jClass
val KFunction<*>.fullSignature: String get() = javaClass.methods.first { it.name == "getSignature" }.apply { isAccessible = true }.invoke(this) as String
val KFunction<*>.method: Method get() = owner.methods.firstOrNull { it.name == name } ?: error("Can't find method $name")

object JvmMemTools {
    @JvmStatic
    fun get8(m: D2Memory, index: Int): Int = m.get8(index)

    @JvmStatic
    fun get16(m: D2Memory, index: Int): Int = m.get16(index)

    @JvmStatic
    fun get32(m: D2Memory, index: Int): Int = m.get32(index)

    @JvmStatic
    fun get64(m: D2Memory, index: Int): Long = m.get64(index)

    @JvmStatic
    fun set8(m: D2Memory, index: Int, value: Int) = m.set8(index, value)

    @JvmStatic
    fun set16(m: D2Memory, index: Int, value: Int) = m.set16(index, value)

    @JvmStatic
    fun set32(m: D2Memory, index: Int, value: Int) = m.set32(index, value)

    @JvmStatic
    fun set64(m: D2Memory, index: Int, value: Long) = m.set64(index, value)

    @JvmStatic
    fun getF32(m: D2Memory, index: Int): Float = Float.fromBits(m.get32(index))

    @JvmStatic
    fun setF32(m: D2Memory, index: Int, value: Float) = m.set32(index, value.toRawBits())

    @JvmStatic
    fun printi(i: Int): Unit = println(i)
}

val <T> Class<T>.internalName get() = Type.getInternalName(this)
val <T : Any> KClass<T>.internalName get() = this.java.internalName

val <T> Class<T>.asmRef get() = "L" + this.internalName + ";"
val <T : Any> KClass<T>.asmRef get() = this.java.asmRef

