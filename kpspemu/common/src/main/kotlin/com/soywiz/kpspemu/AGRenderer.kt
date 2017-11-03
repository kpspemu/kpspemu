package com.soywiz.kpspemu

import com.soywiz.korag.AG
import com.soywiz.korag.shader.*
import com.soywiz.korge.render.RenderContext
import com.soywiz.korge.render.Texture
import com.soywiz.korim.bitmap.Bitmap32
import com.soywiz.korim.color.BGRA
import com.soywiz.korio.stream.openSync
import com.soywiz.korma.Matrix2d
import com.soywiz.kpspemu.ge.*
import com.soywiz.kpspemu.util.PspLogger
import com.soywiz.kpspemu.util.hasFlag
import com.soywiz.kpspemu.util.hex
import com.soywiz.kpspemu.util.setAlpha

class AGRenderer(val emulatorContainer: WithEmulator, val sceneTex: Texture) : WithEmulator by emulatorContainer {
	var directFastSharpRendering = false

	companion object {
		//val RGBA_TO_BGRA = OS.isJs
		val RGBA_TO_BGRA = false
	}

	val logger = PspLogger("AGRenderer")
	val batchesQueue = arrayListOf<List<GeBatch>>()
	val tempBmp = Bitmap32(512, 272)

	data class Stats(
		var batches: Int = 0,
		var vertices: Int = 0
	) {
		fun reset() {
			batches = 0
			vertices = 0
		}

		override fun toString(): String = "Batches: $batches\nVertices: $vertices"
	}

	val stats = Stats()

	private var indexBuffer: AG.Buffer? = null
	private var vertexBuffer: AG.Buffer? = null

	fun render(ctx: RenderContext, m: Matrix2d) {
		val ag = ctx.ag
		ag.checkErrors = false
		ctx.flush()
		stats.reset()

		if (directFastSharpRendering) {
			if (batchesQueue.isNotEmpty()) {
				renderBatches(ag)
			}
		} else {
			if (batchesQueue.isNotEmpty()) {
				mem.read(display.address, tempBmp.data)
				ag.renderToBitmapEx(tempBmp) {
					ag.drawBmp(tempBmp)
					renderBatches(ag)
				}
				mem.write(display.address, tempBmp.data)
			}

			if (display.rawDisplay) {
				display.decodeToBitmap32(display.bmp)
				display.bmp.setAlpha(0xFF)
				sceneTex.update(display.bmp)
			}

			ctx.batch.drawQuad(sceneTex, m = m, blendFactors = AG.Blending.NONE, filtering = false)
		}
		ctx.flush()
	}

	private val renderState = AG.RenderState()
	private val vr = VertexReader()
	private val vv = VertexRaw()

	private fun renderBatches(ag: AG) {
		stats.reset()
		try {
			for (batches in batchesQueue) for (batch in batches) renderBatch(ag, batch)
		} finally {
			batchesQueue.clear()
		}
	}

	private fun renderBatch(ag: AG, batch: GeBatch) {
		stats.batches++
		stats.vertices += batch.vertexCount
		//if (batch.vertexCount < 10) return

		if (indexBuffer == null) indexBuffer = ag.createIndexBuffer()
		if (vertexBuffer == null) vertexBuffer = ag.createVertexBuffer()

		val state = batch.state

		indexBuffer!!.upload(batch.indices)
		vertexBuffer!!.upload(batch.vertices)

		//logger.level = PspLogLevel.TRACE
		logger.trace { "----------------" }
		logger.trace { "indices: ${batch.indices.toList()}" }
		logger.trace { "primitive: ${batch.primType.toAg()}" }
		logger.trace { "vertexCount: ${batch.vertexCount}" }
		logger.trace { "vertexType: ${batch.state.vertexType.hex}" }
		logger.trace { "vertices: ${batch.vertices.hex}" }
		logger.trace { "matrix: ${batch.modelViewProjMatrix}" }

		logger.trace {
			val vr = VertexReader()
			"" + vr.read(batch.vtype, batch.vertices.size / batch.vtype.size(), batch.vertices.openSync())
		}

		if (state.clearing) {
			val fixedDepth = state.depthTest.rangeNear.toFloat()
			renderState.depthNear = fixedDepth
			renderState.depthFar = fixedDepth
			renderState.depthMask = (state.clearFlags hasFlag ClearBufferSet.DepthBuffer)
			renderState.depthFunc = AG.CompareMode.ALWAYS
			//batch.modelViewProjMatrix.setToOrtho(0f, 272f, 480f, 0f, 0f, (-0xFFFF).toFloat())
			batch.modelViewProjMatrix.setToOrtho(0f, 272f, 480f, 0f, (-0xFFFF).toFloat(), 0f)
			val vertex = vr.readOne(batch.vertices.openSync(), batch.vtype, vv)
			ag.clear(
				if (RGBA_TO_BGRA) BGRA.packRGBA(vertex.color) else vertex.color,
				fixedDepth, state.stencil.funcRef,
				clearColor = state.clearFlags hasFlag ClearBufferSet.ColorBuffer,
				clearDepth = state.clearFlags hasFlag ClearBufferSet.DepthBuffer,
				clearStencil = state.clearFlags hasFlag ClearBufferSet.StencilBuffer
			)
			return

		} else {
			// @TODO: Invert this since in PSP it is reversed and WebGL doesn't support it
			renderState.depthNear = state.depthTest.rangeFar.toFloat()
			renderState.depthFar = state.depthTest.rangeNear.toFloat()
			renderState.depthMask = state.depthTest.mask == 0
			renderState.depthFunc = when {
			//state.depthTest.enabled -> state.depthTest.func.toAg()
				state.depthTest.enabled -> state.depthTest.func.toInvAg()
				else -> {
					AG.CompareMode.ALWAYS
				}
			}
		}

		//println(renderState)

		val pl = getProgramLayout(state)
		ag.draw(
			type = batch.primType.toAg(),
			vertices = vertexBuffer!!,
			indices = indexBuffer!!,
			program = pl.program,
			vertexLayout = pl.layout,
			vertexCount = batch.vertexCount,
			uniforms = mapOf(
				u_modelViewProjMatrix to batch.modelViewProjMatrix
			),
			blending = AG.Blending.NONE,
			renderState = renderState
		)

		//val out = FloatArray(512 * 1); ag.readDepth(512, 1, out); println(out.toList())
	}

	val u_modelViewProjMatrix = Uniform("u_modelViewProjMatrix", VarType.Mat4)

	data class ProgramLayout(val program: Program, val layout: VertexLayout)

	val programLayoutByVertexType = LinkedHashMap<Int, ProgramLayout>()

	fun getProgramLayout(state: GeState): ProgramLayout {
		return programLayoutByVertexType.getOrPut(state.vertexType) { createProgramLayout(state) }
	}

	fun createProgramLayout(state: GeState): ProgramLayout {
		val vtype = VertexType().init(state)
		val COUNT2 = listOf(VarType.VOID, VarType.BYTE(2), VarType.SHORT(2), VarType.FLOAT(2))
		val COUNT3 = listOf(VarType.VOID, VarType.BYTE(3), VarType.SHORT(3), VarType.FLOAT(3))

		//val COLORS = listOf(VarType.VOID, VarType.VOID, VarType.VOID, VarType.VOID, VarType.SHORT(1), VarType.SHORT(1), VarType.SHORT(1), VarType.BYTE(4))
		val COLORS = listOf(VarType.VOID, VarType.VOID, VarType.VOID, VarType.VOID, VarType.SHORT(1), VarType.SHORT(1), VarType.SHORT(1), VarType.Byte4)

		//val a_Tex = Attribute("a_Tex", VarType.Float2, normalized = false)
		val a_Tex = if (vtype.hasTexture) Attribute("a_Tex", COUNT2[vtype.tex.id], normalized = false, offset = vtype.texOffset) else null
		val a_Col = if (vtype.hasColor) Attribute("a_Col", COLORS[vtype.col.id], normalized = true, offset = vtype.colOffset) else null
		val a_Pos = Attribute("a_Pos", COUNT3[vtype.pos.id], normalized = false, offset = vtype.posOffset)
		val v_Col = Varying("v_Col", VarType.Byte4)

		val layout = VertexLayout(listOf(a_Tex, a_Col, a_Pos).filterNotNull(), vtype.size())

		val program = Program(
			name = "$vtype",
			vertex = VertexShader {
				//SET(out, vec4(a_Pos, 0f.lit, 1f.lit) * u_ProjMat)
				//SET(out, u_modelViewProjMatrix * vec4(a_Pos, 1f.lit))
				SET(out, u_modelViewProjMatrix * vec4(a_Pos, 1f.lit))
				if (a_Col != null) {
					SET(v_Col, a_Col[if (RGBA_TO_BGRA) "bgra" else "rgba"])
				}
				//SET(out, vec4(a_Pos, 1f.lit))
			},
			fragment = FragmentShader {
				if (a_Col != null) {
					SET(out, v_Col)
				}
			}
		)

		return ProgramLayout(program, layout)
	}
}

private fun PrimitiveType.toAg(): AG.DrawType = when (this) {
	PrimitiveType.POINTS -> AG.DrawType.POINTS
	PrimitiveType.LINES -> AG.DrawType.LINES
	PrimitiveType.LINE_STRIP -> AG.DrawType.LINE_STRIP
	PrimitiveType.TRIANGLES -> AG.DrawType.TRIANGLES
	PrimitiveType.TRIANGLE_STRIP -> AG.DrawType.TRIANGLE_STRIP
	PrimitiveType.TRIANGLE_FAN -> AG.DrawType.TRIANGLE_FAN
	PrimitiveType.SPRITES -> {
		//invalidOp("Can't handle sprite primitives")
		AG.DrawType.TRIANGLES
	}
}

private fun TestFunctionEnum.toAg() = when (this) {
	TestFunctionEnum.NEVER -> AG.CompareMode.NEVER
	TestFunctionEnum.ALWAYS -> AG.CompareMode.ALWAYS
	TestFunctionEnum.EQUAL -> AG.CompareMode.EQUAL
	TestFunctionEnum.NOT_EQUAL -> AG.CompareMode.NOT_EQUAL
	TestFunctionEnum.LESS -> AG.CompareMode.LESS
	TestFunctionEnum.LESS_OR_EQUAL -> AG.CompareMode.LESS_EQUAL
	TestFunctionEnum.GREATER -> AG.CompareMode.GREATER
	TestFunctionEnum.GREATER_OR_EQUAL -> AG.CompareMode.GREATER_EQUAL
}

private fun TestFunctionEnum.toInvAg() = when (this) {
	TestFunctionEnum.NEVER -> AG.CompareMode.NEVER
	TestFunctionEnum.ALWAYS -> AG.CompareMode.ALWAYS
	TestFunctionEnum.EQUAL -> AG.CompareMode.EQUAL
	TestFunctionEnum.NOT_EQUAL -> AG.CompareMode.NOT_EQUAL
	TestFunctionEnum.LESS -> AG.CompareMode.GREATER
	TestFunctionEnum.LESS_OR_EQUAL -> AG.CompareMode.GREATER_EQUAL
	TestFunctionEnum.GREATER -> AG.CompareMode.LESS
	TestFunctionEnum.GREATER_OR_EQUAL -> AG.CompareMode.LESS_EQUAL
}
