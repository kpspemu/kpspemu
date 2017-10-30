package com.soywiz.kpspemu.ge

import com.soywiz.korio.util.extract
import com.soywiz.kpspemu.Emulator
import com.soywiz.kpspemu.WithEmulator
import com.soywiz.kpspemu.callbackManager
import com.soywiz.kpspemu.mem
import com.soywiz.kpspemu.util.ResourceItem
import com.soywiz.kpspemu.util.ResourceList

class Ge(override val emulator: Emulator) : WithEmulator {
	val state = GeState()
	val queue = arrayListOf<GeList>()
	val lists = ResourceList<GeList>("GeList") { GeList(this, it) }

	fun listEnqueue(start: Int, stall: Int, callback: GeCallback, pspGeListArgs: Int): GeList {
		val list = lists.alloc().apply {
			this.start = start
			this.stall = stall
			this.callback = callback
			this.pspGeListArgs = pspGeListArgs
			this.PC = start
			reset()
		}
		queue += list
		return list
	}

	fun run() {
		while (queue.isNotEmpty()) {
			val item = queue.first()
			item.run()
			if (item.completed) {
				lists.free(item)
				queue.removeAt(0)
			} else {
				break
			}
		}
	}

	fun emitBatch(batch: GeBatch) {
		println("BATCH: $batch")
	}

	fun sync(syncType: Int) {
	}
}

data class GeCallback(override val id: Int) : ResourceItem {
	var signal_func: Int = 0
	var signal_arg: Int = 0
	var finish_func: Int = 0
	var finish_arg: Int = 0
}

class GeBatchBuilder(val ge: Ge) {
	var primBatchPrimitiveType: Int = -1
	var primitiveType: Int = -1
	var vertexType: Int = -1
	var vertexCount: Int = 0

	fun reset() {
		primBatchPrimitiveType = -1
		primitiveType = -1
		vertexType = -1
		vertexCount = 0
	}

	fun setVertexKind(primitiveType: Int, vertexType: Int) {
		if (this.primitiveType != primitiveType || this.vertexType != vertexType) flush()
		this.primitiveType = primitiveType
		this.vertexType = vertexType
	}

	fun tflush() {
	}

	fun tsync() {
	}

	fun flush() {
		if (vertexCount > 0) {
			ge.emitBatch(GeBatch(ge.state.clone(), primitiveType, vertexCount, vertexType))
			vertexCount = 0
		}
	}

	fun addVertices(delta: Int) {
		this.vertexCount += delta
	}

}

class GeList(val ge: Ge, override val id: Int) : ResourceItem, WithEmulator by ge {
	var start: Int = 0
	var stall: Int = 0
	var callback: GeCallback = GeCallback(-1)
	var pspGeListArgs: Int = 0
	var PC: Int = start
	var completed: Boolean = false
	val bb = GeBatchBuilder(ge)

	fun reset() {
		completed = false
		bb.reset()
	}

	var callstackIndex = 0
	val callstack = IntArray(0x100)
	val state = ge.state
	val stateData = state.data

	val isStalled: Boolean get() = (stall != 0) && (PC >= stall)

	fun run() {
		val mem = this.mem
		//println("GeList[$id].run: completed=$completed, PC=${PC.hexx}, stall=${stall.hexx}")
		while (!completed && !isStalled) {
			step(mem.lw(PC))
			PC += 4
		}
	}

	fun step(i: Int) {
		val op: Int = i ushr 24
		val p: Int = i and 0xFFFFFF
		when (op) {
			Op.PRIM -> prim(p)
			Op.BEZIER -> {
				println("BEZIER")
			}
			Op.END -> {
				println("END")
				bb.flush()
				completed = true
			}
			Op.TFLUSH -> bb.tflush()
			Op.TSYNC -> bb.tsync()
			Op.NOP -> {
				println("NOP")
			}
			Op.DUMMY -> {
				println("DUMMY")
			}
			Op.JUMP, Op.CALL -> {
				if (op == Op.CALL) {
					callstack[callstackIndex++] = PC + 4
					callstack[callstackIndex++] = (state.baseOffset ushr 2)
				}
				PC = state.baseAddress + p and 0b11.inv()
			}
			Op.RET -> {
				TODO("RET")
			}
			Op.FINISH -> finish(p)
			Op.SIGNAL -> signal(p)
			Op.BASE, Op.IADDR, Op.VADDR, Op.OFFSETADDR -> {
				// Do not invalidate prim
			}
			Op.PROJMATRIXDATA -> state.writeInt(Op.PROJMATRIXNUMBER, Op.MAT_PROJ, p)
			Op.VIEWMATRIXDATA -> state.writeInt(Op.VIEWMATRIXNUMBER, Op.MAT_VIEW, p)
			Op.WORLDMATRIXDATA -> state.writeInt(Op.WORLDMATRIXNUMBER, Op.MAT_WORLD, p)
			Op.BONEMATRIXDATA -> state.writeInt(Op.BONEMATRIXNUMBER, Op.MAT_BONES, p)
			Op.TGENMATRIXDATA -> state.writeInt(Op.TGENMATRIXNUMBER, Op.MAT_TEXTURE, p)

			else -> {
				if (ge.state.data[op] != p) bb.flush()
			}
		}
		stateData[op] = p
	}

	private fun prim(p: Int): PrimAction {
		val primitiveType = p.extract(16, 3)
		val vertexCount: Int = p.extract(0, 16)
		val vertexType: Int = state.vertexType

		bb.setVertexKind(primitiveType, vertexType)
		bb.addVertices(vertexCount)

		println("PRIM: $p -- vertxCount=$vertexCount, primitiveType=$primitiveType, vertexType=$vertexType")
		return PrimAction.FLUSH_PRIM
	}

	private fun finish(p: Int) {
		println("FINISH")
		callbackManager.queueFunction1(callback.finish_func, callback.finish_arg)
		bb.flush()
	}

	private fun signal(p: Int) {
		println("SIGNAL")
		callbackManager.queueFunction1(callback.signal_func, callback.signal_arg)
	}

	fun sync(syncType: Int) {
	}
}

enum class PrimAction { NOTHING, FLUSH_PRIM }

