package com.soywiz.dynarek2.target.jvm

import org.objectweb.asm.*
import org.objectweb.asm.Opcodes.*
import java.lang.reflect.*

fun MethodVisitor.visitPop() = visitInsn(POP)
fun MethodVisitor.visitIReturn() = visitInsn(IRETURN)
fun MethodVisitor.pushBool(value: Boolean) = pushInt(if (value) 1 else 0)
fun MethodVisitor.pushInt(value: Int) {
    when (value) {
        -1 -> visitInsn(ICONST_M1)
        0 -> visitInsn(ICONST_0)
        1 -> visitInsn(ICONST_1)
        2 -> visitInsn(ICONST_2)
        3 -> visitInsn(ICONST_3)
        4 -> visitInsn(ICONST_4)
        5 -> visitInsn(ICONST_5)
        else -> {
            when (value) {
                in -0x80..0x7f -> visitIntInsn(BIPUSH, value)
                in -0x8000..0x7fff -> visitIntInsn(SIPUSH, value)
                else -> visitLdcInsn(value)
            }
        }
    }
}
fun MethodVisitor.pushFloat(value: Float) {
    //visitInsn(FCONST_1)
    visitLdcInsn(value)
}

fun MethodVisitor.pushLong(value: Long) {
    visitLdcInsn(value)
}

val Method.signature: String
    get() {
        val args = this.parameterTypes.map { it.internalName2 }.joinToString("")
        val ret = this.returnType.internalName2
        return "($args)$ret"
    }

val Class<*>.internalName: String get() = this.name.replace('.', '/')
val Class<*>.internalName2: String
    get() = when {
        isPrimitive -> {
            when (this) {
                java.lang.Void.TYPE -> "V"
                java.lang.Integer.TYPE -> "I"
                java.lang.Long.TYPE -> "J"
                java.lang.Float.TYPE -> "F"
                java.lang.Boolean.TYPE -> "Z"
                else -> TODO("Unknown primitive $this")
            }
        }
        isArray -> "[" + this.componentType.internalName2
        else -> "L${this.internalName};"
    }
