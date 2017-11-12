package com.soywiz.dynarek

@Suppress("UNCHECKED_CAST")
class DSlowInterpreter(val args: List<*>) {
	fun <T> interpret(node: DExpr<T>): T = when (node) {
		is DLiteral<*> -> node.value as T
		is DArg<*> -> args[node.index] as T
		is DBinop<*> -> {
			val l = interpret(node.left)
			val r = interpret(node.right)
			when (node.op) {
				"+" -> when {
					l is Int && r is Int -> (l + r) as T
					else -> TODO("Can't ($l + $r)")
				}
				"*" -> when {
					l is Int && r is Int -> (l * r) as T
					else -> TODO("Can't ($l * $r)")
				}
				else -> TODO("Unknown op ${node.op}")
			}
		}
		else -> TODO("Not implemented $node")
	}

	fun interpret(node: DStm): Unit = when (node) {
		is DStms -> for (stm in node.stms) interpret(stm)
		is DAssign<*> -> {
			val left = node.left
			val value = interpret(node.value)
			when (left) {
				is DBindedProp<*, *> -> {
					val obj = interpret(left.obj)
					(left as DBindedProp<Any?, Any?>).prop.set(obj, value)
				}
				else -> TODO("Not implemented left assign ${node.left}")
			}
		}
		else -> TODO("Not implemented $node")
	}

	fun interpret(func: DFunction) {
		interpret(func.body)
	}
}

