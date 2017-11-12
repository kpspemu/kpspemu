package com.soywiz.dynarek

import org.objectweb.asm.ClassWriter
import org.objectweb.asm.MethodVisitor
import org.objectweb.asm.Opcodes.*
import org.objectweb.asm.Type
import java.lang.reflect.Field

var dynarekLastId = 0

val Class<*>.internalName: String get() = this.name.replace('.', '/')
val Class<*>.internalName2: String
	get() = when {
		this == java.lang.Integer.TYPE -> "I"
		this == java.lang.Float.TYPE -> "F"
		isArray -> "[" + this.componentType.internalName2
		else -> "L${this.internalName};"
	}
//fun Field.getDesc() =

fun DFieldAccess<*, *>.getField(): Field {
	val leftClazz = this.clazz.java
	val field = leftClazz.getDeclaredField(prop.name)
	field.isAccessible = true
	return field
}

inline fun log(msgGen: () -> String) {
	//println(msgGen())
}

val jvmOpcodes by lazy { JvmOpcodes.values().map { it.id to it }.toMap() }

fun MethodVisitor._visitInsn(opcode: Int) {
	log { "visitInsn(${jvmOpcodes[opcode]})" }
	visitInsn(opcode)
}

fun MethodVisitor._visitVarInsn(opcode: Int, _var: Int) {
	log { "visitVarInsn(${jvmOpcodes[opcode]}, $_var)" }
	visitVarInsn(opcode, _var)
}

fun MethodVisitor._visitFieldInsn(opcode: Int, owner: String, name: String, desc: String) {
	log { "visitFieldInsn(${jvmOpcodes[opcode]}, $owner, $name, $desc)" }
	visitFieldInsn(opcode, owner, name, desc)
}


fun MethodVisitor._visitTypeInsn(opcode: Int, type: String): Unit {
	log { "visitTypeInsn(${jvmOpcodes[opcode]}, $type)" }
	visitTypeInsn(opcode, type)
}

fun MethodVisitor._visitLdcInsn(cst: Any?): Unit {
	log { "visitLdcInsn($cst)" }
	visitLdcInsn(cst)
}

fun MethodVisitor._visitMethodInsn(opcode: Int, owner: String, name: String, desc: String, itf: Boolean): Unit {
	log { "visitMethodInsn($opcode, $owner, $name, $desc, $itf)" }
	visitMethodInsn(opcode, owner, name, desc, itf)
}

fun MethodVisitor._visitFieldInsn(opcode: Int, field: Field): Unit = _visitFieldInsn(opcode, field.declaringClass.internalName, field.name, field.type.internalName2)
fun MethodVisitor._visitTypeInsn(opcode: Int, type: Class<*>): Unit = _visitTypeInsn(opcode, type.internalName)

fun MethodVisitor._visitCastTo(type: Class<*>): Unit {
	if (type.isPrimitive) {
		when { // unbox
			type.isPrimitiveIntClass() -> {
				_visitTypeInsn(CHECKCAST, "java/lang/Integer")
				_visitMethodInsn(INVOKEVIRTUAL, "java/lang/Integer", "intValue", "()I", false)
			}
		}
	} else {
		_visitTypeInsn(CHECKCAST, type)
	}
}

fun MethodVisitor.visit(expr: DExpr<*>): Unit = when (expr) {
	is DArg<*> -> {
		val aindex = expr.index + 1
		val clazz = expr.clazz
		when (clazz) {
			java.lang.Integer.TYPE.kotlin -> _visitVarInsn(ILOAD, aindex)
			else -> _visitVarInsn(ALOAD, aindex)
		}
	}
	is DBinopInt -> {
		visit(expr.left)
		visit(expr.right)
		when (expr.op) {
			"+" -> _visitInsn(IADD)
			"*" -> _visitInsn(IMUL)
			else -> TODO("Unsupported operator ${expr.op}")
		}
	}
	is DFieldAccess<*, *> -> {
		visit(expr.obj)
		_visitFieldInsn(GETFIELD, expr.getField())
	}
	is DLiteral<*> -> {
		val value = expr.value
		when (value) {
			is Int -> {
				//_visitInsn(ICONST_1)
				_visitLdcInsn(value)
			}
			else -> TODO("MethodVisitor.visit: $expr")
		}
	}
	else -> TODO("MethodVisitor.visit: $expr")
}

fun MethodVisitor.visit(stm: DStm): Unit = when (stm) {
	is DAssign<*> -> {
		val left = stm.left
		val right = stm.value
		when (left) {
			is DFieldAccess<*, *> -> {
				visit(left.obj)
				visit(right)
				_visitFieldInsn(PUTFIELD, left.getField())
			}
			else -> TODO("MethodVisitor.visit.DAssign: $left, $right")
		}
	}
	is DStms -> {
		for (s in stm.stms) visit(s)
	}
	else -> TODO("MethodVisitor.visit: $stm")
}

fun MethodVisitor.visit(func: DFunction) {
	visit(func.body)
}

fun Class<*>.isPrimitiveIntClass() = this == java.lang.Integer.TYPE

fun <T> _generateDynarek(func: DFunction, interfaceClass: Class<T>): T {
	val nargs = func.args.size
	val classId = dynarekLastId++
	val cw = ClassWriter(0)
	val className = "com/soywiz/dynarek/Generated$classId"
	val refObj = "Ljava/lang/Object;"
	val refObjArgs = (0 until nargs).map { refObj }.joinToString("")
	cw.visit(V1_5, ACC_PUBLIC or ACC_FINAL, className, null, "java/lang/Object", arrayOf(interfaceClass.canonicalName.replace('.', '/')))

	val typedArgSignature = func.args.map { it.clazz.java.internalName2 }.joinToString("")
	val typedRetSignature = func.ret.clazz.java.internalName2
	//println(typedArgSignature)
	//println(typedRetSignature)

	cw.apply {
		// constructor
		visitMethod(ACC_PUBLIC, "<init>", "()V", null, null).apply {
			visitMaxs(2, 1)
			visitVarInsn(ALOAD, 0) // push `this` to the operand stack
			visitMethodInsn(INVOKESPECIAL, Type.getInternalName(Any::class.java), "<init>", "()V", false) // call the constructor of super class
			visitInsn(RETURN)
			visitEnd()
		}
		// invoke method
		visitMethod(ACC_PUBLIC, "invoke", "($typedArgSignature)$typedRetSignature", null, null).apply {
			log { "-------- invoke($typedArgSignature)$typedRetSignature" }
			visitMaxs(16, 16)
			visit(func)
			visitInsn(ACONST_NULL)
			visitInsn(ARETURN)
			visitEnd()
		}
		visitMethod(ACC_PUBLIC, "invoke", "($refObjArgs)$refObj", null, null).apply {
			log { "-------- invoke($refObjArgs)$refObj\", null, null)" }
			visitMaxs(16, 16)
			_visitVarInsn(ALOAD, 0) // this
			for ((index, arg) in func.args.withIndex()) {
				val jclazz = arg.clazz.java
				_visitVarInsn(ALOAD, index + 1) // this
				_visitCastTo(jclazz)
			}
			_visitMethodInsn(INVOKEVIRTUAL, className, "invoke", "($typedArgSignature)$typedRetSignature", false)
			//visitInsn(POP)
			//visitInsn(ACONST_NULL)
			visitInsn(ARETURN)
			visitEnd()
		}
	}
	cw.visitEnd()
	val classBytes = cw.toByteArray()

	val gclazz = createDynamicClass(ClassLoader.getSystemClassLoader(), className.replace('/', '.'), classBytes)
	return gclazz.declaredConstructors.first().newInstance() as T
}

fun createDynamicClass(parent: ClassLoader, clazzName: String, b: ByteArray): Class<*> = OwnClassLoader(parent).defineClass(clazzName, b)

private class OwnClassLoader(parent: ClassLoader) : ClassLoader(parent) {
	fun defineClass(name: String, b: ByteArray): Class<*> = defineClass(name, b, 0, b.size)
}
