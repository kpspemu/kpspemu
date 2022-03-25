package com.soywiz.kpspemu

import com.soywiz.klogger.*
import com.soywiz.kmem.*
import com.soywiz.korag.*
import com.soywiz.korag.shader.*
import com.soywiz.korge.render.*
import com.soywiz.korge.view.*
import com.soywiz.korim.bitmap.*
import com.soywiz.korim.color.*
import com.soywiz.korio.stream.*
import com.soywiz.korma.*
import com.soywiz.kpspemu.ge.*
import com.soywiz.kpspemu.mem.*
import com.soywiz.kpspemu.util.*
import com.soywiz.krypto.encoding.*

fun Bitmap32.setAlpha2(value: Int) {
    for (n in 0 until this.data.size) this.data[n] = RGBA(this.data[n].rgb, value)
}


class AGRenderer(val emulatorContainer: WithEmulator, val sceneTex: Texture) : WithEmulator by emulatorContainer {
    enum class RenderMode { AUTO, NORMAL, DIRECT }

    var renderMode = RenderMode.AUTO

    val logger = Logger("AGRenderer")
    var anyBatch = false
    private val batchesQueue = arrayListOf<List<GeBatchData>>()
    val tempBmp = Bitmap32(480, 272)
    val taskCountFlag = EventFlag(0)
    var taskCount by taskCountFlag

    fun reset() {
        for ((_, tex) in texturesById) tex.texture.close()
        for ((_, vtype) in programLayoutByVertexType) {
            //vtype.dele
            // close
        }
        texturesById.clear()
        programLayoutByVertexType.clear()
        taskCount = 0
    }

    fun addBatches(batches: List<GeBatchData>) {
        anyBatch = true
        batchesQueue += batches
        stats.add(batches)
        updateTaskCount()
    }

    private fun updateTaskCount() {
        taskCount = batchesQueue.size
    }

    val stats = Stats()

    private var indexBuffer: AG.Buffer? = null
    private var vertexBuffer: AG.Buffer? = null

    var renderBuffer: AG.RenderBuffer? = null
    lateinit var geTexture: Texture

    var renderScale = 2.0

    var frameCount = 0

    //val WW = 512
    val WW = 480
    val HH = 272
    var actualTexture = sceneTex

    fun render(views: Views, ctx: RenderContext, m: Matrix2d) {
        val ag = ctx.ag
        ctx.flush()
        frameCount++

        val directFastSharpRendering = when (renderMode) {
            RenderMode.AUTO -> anyBatch
            RenderMode.NORMAL -> false
            RenderMode.DIRECT -> true
        }

        if (directFastSharpRendering) {

            if (batchesQueue.isNotEmpty()) {
                //ag.create

                if (renderBuffer == null) {
                    renderBuffer = ag.createRenderBuffer()
                    geTexture = Texture(TextureBase(renderBuffer!!.tex, WW, HH), 0, HH, WW, 0)
                    //geTexture = Texture(Texture.Base(renderBuffer!!.tex, WW, HH))
                    renderBuffer!!.setSize(0, 0, WW * renderScale.toInt(), HH * renderScale.toInt())
                    ag.setRenderBufferTemporally(renderBuffer!!) {
                        ag.clear(Colors.BLACK)
                    }
                }

                val rb = renderBuffer!!
                rb.setSize(0, 0, WW * renderScale.toInt(), HH * renderScale.toInt())
                ag.setRenderBufferTemporally(rb) {
                    //ag.clear() // @TODO: Is this required?
                    renderBatches(views, ctx, scale = renderScale)
                }
            }
            actualTexture = geTexture
        } else {
            if (batchesQueue.isNotEmpty()) {
                //mem.read(display.fixedAddress(), tempBmp.data)
                for (n in 0 until 272) {
                    //mem.write(display.fixedAddress() + n * 512 * 4, tempBmp.data, tempBmp.width * (271 - n), tempBmp.width)
                    mem.read(display.fixedAddress() + n * 512 * 4, tempBmp.data.ints, tempBmp.width * n, tempBmp.width)
                }
                ag.renderToBitmap(tempBmp) {
                    ag.drawBitmap(tempBmp)
                    renderBatches(views, ctx, scale = 1.0)
                }
                for (n in 0 until 272) {
                    mem.write(
                        display.fixedAddress() + n * 512 * 4,
                        tempBmp.data.ints,
                        tempBmp.width * (271 - n),
                        tempBmp.width
                    )
                }
            }

            if (display.rawDisplay) {
                display.decodeToBitmap32(display.bmp)
                display.bmp.setAlpha2(0xFF)
                sceneTex.update(display.bmp)
            }
            actualTexture = sceneTex
        }
        ctx.flush()
    }

    fun renderTexture(views: Views, ctx: RenderContext, m: Matrix2d) {
        ctx.flush()
        ctx.batch.drawQuad(
            actualTexture,
            m = m,
            blendFactors = AG.Blending.NONE,
            filtering = false,
            width = WW.toFloat(),
            height = HH.toFloat()
        )
        ctx.flush()
    }

    private val renderState = AG.RenderState()
    private val vr = VertexReader()
    private val vv = VertexRaw()
    private val batch = GeBatch()
    val u_modelViewProjMatrix = Uniform("u_modelViewProjMatrix", VarType.Mat4)
    val u_tex = Uniform("u_tex", VarType.TextureUnit)
    val u_texMatrix = Uniform("u_texMatrix", VarType.Mat4)
    val textureMatrix = Matrix4()
    val textureUnit = AG.TextureUnit(null, linear = false)
    val u_Col = Uniform("u_Col", VarType.Float4)
    val color = FloatArray(4)

    private val uniforms = AG.UniformValues(
        u_modelViewProjMatrix to batch.modelViewProjMatrix,
        u_tex to textureUnit,
        u_texMatrix to textureMatrix,
        u_Col to color
    )

    private fun renderBatches(views: Views, ctx: RenderContext, scale: Double) {
        try {
            for (batches in batchesQueue) for (batch in batches) {
                this.batch.initData(batch, emulatorContainer.emulator)
                renderBatch(views, ctx, this.batch, scale)
            }
        } finally {
            batchesQueue.clear()
        }
        updateTaskCount()
    }

    data class TextureSlot(
        val id: Int,
        val texture: AG.Texture,
        var version: Int = 0,
        var frame: Int = 0,
        var hash: Int = 0
    )

    val vtype = VertexType()
    //var texture: AG.Texture? = null
    val texturesById = LinkedHashMap<Int, TextureSlot>()
    //val texturesById = IntMap<TextureSlot>()

    private fun renderBatch(views: Views, ctx: RenderContext, batch: GeBatch, scale: Double) {
        val ag = ctx.ag
        //if (texture == null) {
        //	texture = ag.createTexture()
        //}
        vtype.init(batch.state)

        //if (batch.vertexCount < 10) return

        if (indexBuffer == null) indexBuffer = ag.createIndexBuffer()
        if (vertexBuffer == null) vertexBuffer = ag.createVertexBuffer()

        val state = batch.state

        indexBuffer!!.upload(batch.indices)
        vertexBuffer!!.upload(batch.vertices)


        //logger.level = LogLevel.TRACE
        logger.trace { "----------------" }
        logger.trace { "indices: ${batch.indices.toList()}" }
        logger.trace { "primitive: ${batch.primType.toAg()}" }
        logger.trace { "vertexCount: ${batch.vertexCount}" }
        logger.trace { "vertexType: ${batch.state.vertexType.hex}" }
        logger.trace { "vertices: ${batch.vertices.hex}" }
        logger.trace { "matrix: ${batch.modelViewProjMatrix}" }

        logger.trace {
            val vr = VertexReader()
            "" + vr.read(batch.vtype, batch.vertices.size / batch.vtype.size, batch.vertices.openSync())
        }

        logger.trace {
            //val tex = batch.getTextureBitmap(mem)
            //"texture: ${batch.hasTexture()} : ${batch.getTextureId()} : $tex : ${batch.state.texture.mipmap.address.hex} : ${tex?.data?.toList()}"
            "texture: ${batch.hasTexture()} : ${batch.getTextureId()} : ${batch.state.texture.mipmap.address.hex}"
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
            val clearFlags = state.clearFlags
            val clearColor = vertex.color
            val clearDepth = fixedDepth
            val clearStencil = state.stencil.funcRef
            val mustClearColor = clearFlags hasFlag ClearBufferSet.ColorBuffer
            //val mustClearColor = true
            val mustClearDepth = clearFlags hasFlag ClearBufferSet.DepthBuffer
            val mustClearStencil = clearFlags hasFlag ClearBufferSet.StencilBuffer

            ag.clear(
                RGBA(clearColor),
                clearDepth,
                clearStencil,
                clearColor = mustClearColor,
                clearDepth = mustClearDepth,
                clearStencil = mustClearStencil
            )

            return
        }

        // @TODO: Invert this since in PSP it is reversed and WebGL doesn't support it
        renderState.depthNear = state.depthTest.rangeFar.toFloat()
        renderState.depthFar = state.depthTest.rangeNear.toFloat()
        renderState.depthMask = state.depthTest.mask == 0
        renderState.depthFunc = when {
        //state.depthTest.enabled -> state.depthTest.func.toAg()
            state.depthTest.enabled -> state.depthTest.func.toInvAg()
            else -> AG.CompareMode.ALWAYS
        }

        //val nativeWidth = if (direct) views.nativeWidth else 480

        //renderState.lineWidth = nativeWidth.toFloat() / 480.toFloat()
        renderState.lineWidth = scale.toFloat()

        //println("${views.nativeWidth}x${views.nativeHeight}")

        val textureSlot: TextureSlot?

        if (batch.hasTexture()) {
            val textureId = batch.getTextureId()

            textureSlot = texturesById.getOrPut(textureId) {
                TextureSlot(textureId, ag.createTexture(premultiplied = false))
            }

            if (textureSlot.frame != frameCount) {
                textureSlot.frame = frameCount
                val texVersion = batch.data.texVersion
                //val texVersion = 1
                //val texVersion = frameCount

                if (textureSlot.version != texVersion) {
                    textureSlot.version = texVersion

                    val texHash = batch.getTextureHash(mem)

                    if (textureSlot.hash != texHash) {
                        textureSlot.hash = texHash
                        val bmp = batch.getTextureBitmap(mem)
                        textureSlot.texture.upload(bmp)
                        //go(coroutineContext) {
                        //	bmp?.setAlpha(0xFF)
                        //	bmp?.writeTo(LocalVfs("c:/temp/$textureId.png"), formats = PNG)
                        //}
                        //println("Texture upload!")
                    }
                }
            }

        } else {
            textureSlot = null
        }

        batch.getEffectiveTextureMatrix(textureMatrix)

        val blending = when (state.blending.enabled) {
            false -> AG.Blending.NONE
            true -> AG.Blending(
                state.blending.functionSource.toAg(),
                state.blending.functionDestination.toAg(),
                state.blending.functionSource.toAg(),
                state.blending.functionDestination.toAg(),
                state.blending.equation.toAg(),
                state.blending.equation.toAg()
            )
        }

        val texture: AG.Texture? = textureSlot?.texture

        textureUnit.texture = texture
        textureUnit.linear = !state.texture.filterMinification.nearest
        if (vtype.hasColor) {
            color[0] = 1f
            color[1] = 1f
            color[2] = 1f
            color[3] = 1f
        } else {
            color[0] = state.ambientModelColor.r
            color[1] = state.ambientModelColor.g
            color[2] = state.ambientModelColor.b
            color[3] = state.ambientModelColor.a
        }
        //println(state.blending.functionSource)
        //println(state.blending.functionDestination)

        //println(renderState)

        val pl = getProgramLayout(state)
        ag.draw(
            type = batch.primType.toAg(),
            vertices = vertexBuffer!!,
            indices = indexBuffer!!,
            program = pl.program,
            vertexLayout = pl.layout,
            vertexCount = batch.vertexCount,
            uniforms = uniforms,
            blending = blending,
            renderState = renderState
        )

        //val out = FloatArray(512 * 1); ag.readDepth(512, 1, out); println(out.toList())
    }

    data class ProgramLayout(val program: Program, val layout: VertexLayout)

    val programLayoutByVertexType = LinkedHashMap<String, ProgramLayout>()

    fun getProgramLayout(state: GeState): ProgramLayout {
        val hash =
            "" + state.vertexType + "_" + state.texture.effect.id + "_" + state.alphaTest.hash + "_" + state.texture.hasAlpha + "_" + state.texture.effect + "_" + state.alphaTest.enabled
        return programLayoutByVertexType.getOrPut(hash) {
            logger.warn { "getProgramLayout[new]: $hash" }
            createProgramLayout(state)
        }
    }

    private val Operand.xy: Operand get() = Program.Swizzle(this, "xy")
    private val Operand.rgb: Operand get() = Program.Swizzle(this, "rgb")
    private val Operand.a: Operand get() = Program.Swizzle(this, "a")
    private val Operand.rgba: Operand get() = Program.Swizzle(this, "rgba")

    fun createProgramLayout(state: GeState): ProgramLayout {
        val vtype = VertexType().init(state)
        val COUNT2 = listOf(VarType.VOID, VarType.BYTE(2), VarType.SHORT(2), VarType.FLOAT(2))
        val COUNT3 = listOf(VarType.VOID, VarType.BYTE(3), VarType.SHORT(3), VarType.FLOAT(3))

        //val COLORS = listOf(VarType.VOID, VarType.VOID, VarType.VOID, VarType.VOID, VarType.SHORT(1), VarType.SHORT(1), VarType.SHORT(1), VarType.BYTE(4))
        val COLORS = listOf(
            VarType.VOID,
            VarType.VOID,
            VarType.VOID,
            VarType.VOID,
            VarType.SHORT(1),
            VarType.SHORT(1),
            VarType.SHORT(1),
            VarType.Byte4
        )

        //val a_Tex = Attribute("a_Tex", VarType.Float2, normalized = false)
        val a_Tex = if (vtype.hasTexture) Attribute(
            "a_Tex",
            COUNT2[vtype.tex.id],
            normalized = false,
            offset = vtype.texOffset
        ) else null
        val a_Col = if (vtype.hasColor) Attribute(
            "a_Col",
            COLORS[vtype.col.id],
            normalized = true,
            offset = vtype.colOffset
        ) else null
        val a_Pos = Attribute("a_Pos", COUNT3[vtype.pos.id], normalized = false, offset = vtype.posOffset)
        val v_Tex = Varying("v_Tex", VarType.Float4)
        val v_Col = Varying("v_Col", VarType.Byte4)
        val t_Col = Temp(0, VarType.Byte4)
        val layout = VertexLayout(listOf(a_Tex, a_Col, a_Pos).filterNotNull(), vtype.size)
        //println("ambientModelColor:" + state.ambientModelColor)
        //println("diffuseModelColor:" + state.diffuseModelColor)
        //println("specularModelColor:" + state.specularModelColor)

        val program = Program(
            name = "$vtype",
            vertex = VertexShader {
                SET(out, u_modelViewProjMatrix * vec4(a_Pos, 1f.lit))
                if (a_Col != null) {
                    SET(v_Col, a_Col.rgba)
                }
                if (a_Tex != null) {
                    SET(v_Tex, u_texMatrix * vec4(a_Tex.xy, 0f.lit, 0f.lit))
                }
            },
            fragment = FragmentShader {
                SET(out, u_Col)

                if (a_Col != null) {
                    SET(out, out * v_Col)
                }

                if (a_Tex != null) {
                    SET(t_Col, texture2D(u_tex, v_Tex["xy"]))
                    val hasAlpha = state.texture.hasAlpha
                    when (state.texture.effect) {
                        TextureEffect.MODULATE -> {
                            SET(out.rgb, out.rgb * t_Col.rgb)
                            if (hasAlpha) SET(out.a, out.a * t_Col.a)
                        }
                        TextureEffect.DECAL -> {
                            if (hasAlpha) {
                                SET(out.rgb, out.rgb * t_Col.rgb)
                                SET(out.a, t_Col.a)
                            } else {
                                SET(out.rgba, t_Col.rgba)
                            }
                        }
                        TextureEffect.BLEND -> SET(out, mix(out, t_Col, 0.5f.lit))
                        TextureEffect.REPLACE -> {
                            SET(out.rgb, t_Col.rgb)
                            if (hasAlpha) SET(out.a, t_Col.a)
                        }
                        TextureEffect.ADD -> {
                            SET(out.rgb, out.rgb + t_Col.rgb)
                            if (hasAlpha) SET(out.a, out.a * t_Col.a)
                        }
                        else -> Unit
                    }
                }

                if (state.alphaTest.enabled) {
                    IF(out.a le 0f.lit) {
                        DISCARD()
                    }
                }
            }
        )

        return ProgramLayout(program, layout)
    }

    data class Stats(
        var batches: Int = 0,
        var vertices: Int = 0,

        var batchesPoints: Int = 0,
        var batchesLines: Int = 0,
        var batchesTriangles: Int = 0,
        var batchesSprites: Int = 0,

        var verticesPoints: Int = 0,
        var verticesLines: Int = 0,
        var verticesTriangles: Int = 0,
        var verticesSprites: Int = 0,

        var cpuTime: Int = 0,
        var renderTime: Int = 0
    ) {
        fun reset() {
            batches = 0
            vertices = 0

            batchesPoints = 0
            batchesLines = 0
            batchesTriangles = 0
            batchesSprites = 0

            verticesPoints = 0
            verticesLines = 0
            verticesTriangles = 0
            verticesSprites = 0
        }

        override fun toString(): String {
            val lines = arrayListOf<String>()
            lines += "Total: $vertices ($batches)"
            lines += "Points: $verticesPoints ($batchesPoints)"
            lines += "Lines: $verticesLines ($batchesLines)"
            lines += "Triangles: $verticesTriangles ($batchesTriangles)"
            lines += "Sprites: $verticesSprites ($batchesSprites)"
            lines += ""
            lines += "CpuTime: $cpuTime"
            lines += "RenderTime: $renderTime"
            return lines.joinToString("\n")
        }

        fun add(batches: List<GeBatchData>) {
            for (batch in batches) {
                this.batches++
                vertices += batch.vertexCount
                when (batch.primType) {
                    PrimitiveType.POINTS -> {
                        batchesPoints++
                        verticesPoints += batch.vertexCount
                    }
                    PrimitiveType.LINES, PrimitiveType.LINE_STRIP -> {
                        batchesLines++
                        verticesLines += batch.vertexCount
                    }
                    PrimitiveType.TRIANGLES, PrimitiveType.TRIANGLE_STRIP, PrimitiveType.TRIANGLE_FAN -> {
                        batchesTriangles++
                        verticesTriangles += batch.vertexCount
                    }
                    PrimitiveType.SPRITES -> {
                        batchesSprites++
                        verticesSprites += batch.vertexCount
                    }
                }
            }
        }
    }
}

private fun GuBlendingEquation.toAg(): AG.BlendEquation = when (this) {
    GuBlendingEquation.ADD -> AG.BlendEquation.ADD
    GuBlendingEquation.SUBSTRACT -> AG.BlendEquation.SUBTRACT
    GuBlendingEquation.REVERSE_SUBSTRACT -> AG.BlendEquation.REVERSE_SUBTRACT
    GuBlendingEquation.MIN -> AG.BlendEquation.ADD // TODO: Not defined in OpenGL
    GuBlendingEquation.MAX -> AG.BlendEquation.ADD // TODO: Not defined in OpenGL
    GuBlendingEquation.ABS -> AG.BlendEquation.ADD // TODO: Not defined in OpenGL
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

private fun GuBlendingFactor.toAg() = when (this) {
    GuBlendingFactor.GU_SRC_COLOR -> AG.BlendFactor.SOURCE_COLOR
    GuBlendingFactor.GU_ONE_MINUS_SRC_COLOR -> AG.BlendFactor.ONE_MINUS_SOURCE_COLOR
    GuBlendingFactor.GU_SRC_ALPHA -> AG.BlendFactor.SOURCE_ALPHA
    GuBlendingFactor.GU_ONE_MINUS_SRC_ALPHA -> AG.BlendFactor.ONE_MINUS_SOURCE_ALPHA
    GuBlendingFactor.GU_DST_ALPHA -> AG.BlendFactor.DESTINATION_ALPHA
    GuBlendingFactor.GU_ONE_MINUS_DST_ALPHA -> AG.BlendFactor.ONE_MINUS_DESTINATION_ALPHA
    GuBlendingFactor.GU_FIX -> AG.BlendFactor.SOURCE_COLOR
}
