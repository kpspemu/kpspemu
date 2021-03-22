package com.soywiz.kpspemu.ge

import com.soywiz.kds.*
import com.soywiz.kmem.*
import com.soywiz.korio.async.*
import com.soywiz.kpspemu.*
import com.soywiz.kpspemu.mem.*
import com.soywiz.kpspemu.util.*
import com.soywiz.krypto.encoding.*
import kotlin.math.*

class GeList(val ge: Ge, override val id: Int) : ResourceItem, WithEmulator by ge {
    val logger = com.soywiz.klogger.Logger("GeList")

    var start: Int = 0
    var stall: Int = 0
    var callback: GeCallback = GeCallback(-1)
    var pspGeListArgs: Int = 0
    var PC: Int = start
    var completed: Boolean = false
    val bb = GeBatchBuilder(ge)
    var onCompleted = Signal<Unit>()
    var phase = ListSyncKind.QUEUED

    fun reset() {
        completed = false
        bb.reset()
        onCompleted = Signal<Unit>()
        phase = ListSyncKind.QUEUED
    }

    var callstackIndex = 0
    val callstack = IntArray(0x100)
    val state = ge.state
    val stateData = state.data

    val isStalled: Boolean get() = (stall != 0) && (PC >= stall)

    fun run() {
        val mem = this.mem
        PC = PC and MemoryInfo.MASK
        stall = stall and MemoryInfo.MASK
        //println("GeList[$id].run: completed=$completed, PC=${PC.hexx}, stall=${stall.hexx}")
        while (!completed && !isStalled) {
            val cPC = PC
            PC += 4
            step(cPC, mem.lw(cPC))
        }
        if (isStalled) phase = ListSyncKind.STALL_REACHED
        if (completed) {
            phase = ListSyncKind.DRAWING_DONE
            onCompleted(Unit)
        }
    }

    fun step(cPC: Int, i: Int) {
        val op: Int = i ushr 24
        val p: Int = i and 0xFFFFFF
        //logger.level = PspLogLevel.TRACE
        logger.trace { "GE: ${cPC.hex}-${stall.hex}: ${op.hex}" }
        when (op) {
            Op.PRIM -> prim(p)
            Op.TRXKICK -> {
                println("TRXKICK")
            }
            Op.TRXSPOS -> {
                println("TRXSPOS")
            }
            Op.TRXDPOS -> {
                println("TRXDPOS")
            }
            Op.BEZIER -> bezier(p)
            Op.SPLINE -> {
                logger.error { "Not implemented SPLINE" }
            }
            Op.END -> {
                //println("END")
                bb.flush()
                completed = true
            }
            Op.TFLUSH -> {
                bb.tflush()
                bb.flush()
            }
            Op.TSYNC -> bb.tsync()
            Op.NOP -> {
                //println("GE: NOP")
                Unit
            }
            Op.DUMMY -> {
                //println("GE: DUMMY")
                Unit
            }
            Op.JUMP, Op.CALL -> {
                if (op == Op.CALL) {
                    callstack[callstackIndex++] = PC
                    callstack[callstackIndex++] = (state.baseOffset ushr 2)
                }
                PC = (state.baseAddress + p and 0b11.inv()) and MemoryInfo.MASK
            }
            Op.RET -> {
                state.baseOffset = callstack[--callstackIndex] shl 2
                PC = callstack[--callstackIndex]
            }
            Op.FINISH -> finish(p)
            Op.SIGNAL -> signal(p)
            Op.BASE, Op.IADDR, Op.VADDR, Op.OFFSETADDR -> Unit // Do not invalidate prim
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

    private val tempVertices = Array(4 * 4) { VertexRaw() }

    private fun bezier(p: Int) {
        // @TODO: Generate intermediate vertices
        val vt = VertexType(state.vertexType)
        val vr = VertexReader()
        val ucount = max(p.extract8(0), 4) // X
        val vcount = max(p.extract8(8), 4) // Y
        val divs = state.patch.divs // number of divisions X
        val divt = state.patch.divt // number of divisions Y
        val useTexture = vt.hasTexture || state.texture.enabled
        val generateUV = useTexture && !vt.hasTexture

        // 0.1.2.3
        // 4.5.6.7
        // 8.9.A.B
        // C.D.E.F

        val vertices = vr.read(vt, ucount * vcount, mem.getPointerStream(state.vertexAddress), tempVertices)
        val indices = IntArrayList()
        val vertexCount = ucount * vcount

        if (generateUV) {
            var n = 0
            //val coefsU = bernsteinCoeff(1f)

            for (v in 0 until ucount) {
                for (u in 0 until vcount) {
                    val vertex = vertices[n]
                    vertex.tex[0] = u.toFloat() / ((vcount - 1).toFloat())
                    vertex.tex[1] = v.toFloat() / ((ucount - 1).toFloat())
                    //println("${vertex.tex[0]},${vertex.tex[1]}")
                    n++
                }
            }
        }

        run {
            for (u in 0 until ucount - 1) {
                for (v in 0 until vcount - 1) {
                    val n = u * ucount + v
                    indices.add(n + 0)
                    indices.add(n + 1)
                    indices.add(n + ucount)
                    indices.add(n + ucount)
                    indices.add(n + 1)
                    indices.add(n + ucount + 1)
                }
            }
        }
        bb.flush()
        bb.addUnoptimizedShape(
            PrimitiveType.TRIANGLES,
            indices.copyOfShortArray(),
            vertices,
            vertexCount,
            hasPosition = true,
            hasColor = false,
            hasTexture = useTexture,
            hasNormal = false,
            hasWeights = false
        )
    }

    // @TODO: Unused yet! Needed for bezier
    //private fun bernsteinCoeff(u: Float, out: FloatArray = FloatArray(4)): FloatArray {
    //	val uPow2 = u * u
    //	val uPow3 = uPow2 * u
    //	val u1 = 1 - u
    //	val u1Pow2 = u1 * u1
    //	val u1Pow3 = u1Pow2 * u1
    //	out[0] = u1Pow3
    //	out[1] = 3f * u * u1Pow2
    //	out[2] = 3f * uPow2 * u1
    //	out[3] = uPow3
    //	return out
    //}

    private fun prim(p: Int) {
        val primitiveType = PrimitiveType(p.extract(16, 3))
        val vertexCount: Int = p.extract(0, 16)
        //println("PRIM: $primitiveType, $vertexCount")
        bb.setVertexKind(primitiveType, state)
        bb.addIndices(vertexCount)
    }

    private fun finish(p: Int) {
        //println("FINISH")
        callbackManager.queueFunction1(callback.finish_func, callback.finish_arg)
        bb.flush()
    }

    private fun signal(p: Int) {
        //println("SIGNAL")
        callbackManager.queueFunction1(callback.signal_func, callback.signal_arg)
    }

    fun sync(syncType: Int) {
        //println("syncType:$syncType")
        run()
    }

    //fun syncAsync(syncType: Int): Promise<Unit> {
    //	//println("syncType:$syncType")
    //	val deferred = Promise.Deferred<Unit>()
    //	onCompleted.once { deferred.resolve(Unit) }
    //	return deferred.promise
    //}
}
