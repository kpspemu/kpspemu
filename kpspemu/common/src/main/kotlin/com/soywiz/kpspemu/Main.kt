package com.soywiz.kpspemu

import com.soywiz.korag.AG
import com.soywiz.korag.shader.*
import com.soywiz.korge.Korge
import com.soywiz.korge.render.RenderContext
import com.soywiz.korge.scene.Module
import com.soywiz.korge.scene.Scene
import com.soywiz.korge.view.Container
import com.soywiz.korge.view.View
import com.soywiz.korge.view.image
import com.soywiz.korge.view.texture
import com.soywiz.korim.bitmap.Bitmap32
import com.soywiz.korio.JvmStatic
import com.soywiz.korio.inject.AsyncInjector
import com.soywiz.korio.lang.printStackTrace
import com.soywiz.korio.lang.use
import com.soywiz.korio.stream.openSync
import com.soywiz.korio.util.OS
import com.soywiz.korio.vfs.applicationVfs
import com.soywiz.korma.Matrix2d
import com.soywiz.korma.geom.SizeInt
import com.soywiz.kpspemu.format.elf.loadElfAndSetRegisters
import com.soywiz.kpspemu.ge.*
import com.soywiz.kpspemu.hle.registerNativeModules
import com.soywiz.kpspemu.mem.Memory
import kotlin.reflect.KClass

fun main(args: Array<String>) = Main.main(args)

object Main {
	@JvmStatic
	fun main(args: Array<String>) = Korge(KpspemuModule, injector = AsyncInjector()
		.mapPrototype(KpspemuMainScene::class) { KpspemuMainScene() }
	)
}

object KpspemuModule : Module() {
	override val mainScene: KClass<out Scene> = KpspemuMainScene::class
	override val title: String = "kpspemu"
	override val size: SizeInt get() = SizeInt(480, 272)
}

class KpspemuMainScene : Scene(), WithEmulator {
	lateinit override var emulator: Emulator

	class KorgeRenderer(val scene: KpspemuMainScene) : View(scene.views), GpuRenderer, WithEmulator {
		override val emulator: Emulator get() = scene.emulator
		val batchesQueue = arrayListOf<List<GeBatch>>()

		override fun render(batches: List<GeBatch>) {
			batchesQueue += batches
			//println("KorgeRenderer.render: $batches")
			//display.rawDisplay = false
		}

		val tempBmp = Bitmap32(512, 272)

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
			val a_Tex = if (vtype.hasTexture) Attribute("a_Tex", COUNT2[vtype.texture.id], normalized = false) else null
			val a_Col = if (vtype.hasColor) Attribute("a_Col", COLORS[vtype.color.id], normalized = false) else null
			val a_Pos = Attribute("a_Pos", COUNT3[vtype.position.id], normalized = false)

			val v_Col = Varying("v_Col", VarType.Byte4)

			val layout = VertexLayout(listOf(a_Tex, a_Col, a_Pos).filterNotNull())

			val program = Program(
				vertex = VertexShader {
					//SET(out, vec4(a_Pos, 0f.lit, 1f.lit) * u_ProjMat)
					SET(out, u_modelViewProjMatrix * vec4(a_Pos, 1f.lit))
					if (a_Col != null) {
						SET(v_Col, a_Col)
					}
					//SET(out, vec4(a_Pos, 1f.lit))
				},
				fragment = FragmentShader {
					if (a_Col != null) {
						SET(out, v_Col)
					}
				},
				name = "PROGRAM_DEBUG"
			)

			return ProgramLayout(program, layout)
		}

		override fun render(ctx: RenderContext, m: Matrix2d) {
			if (batchesQueue.isNotEmpty()) {
				try {
					val ag = ctx.ag
					//mem.read(display.address, tempBmp.data)
					ag.renderToBitmap(tempBmp) {
						//ag.clear(Colors.BLUE) // @TODO: Remove this

						for (batches in batchesQueue) {
							for (batch in batches) {
								ag.createIndexBuffer().use { indexBuffer ->
									ag.createVertexBuffer().use { vertexBuffer ->
										indexBuffer.upload(batch.indices)
										vertexBuffer.upload(batch.vertices)

										//println("----------------")
										//println("indices: ${batch.indices.toList()}")
										//println("vertices: ${batch.vertices.toList()}")
										//println("matrix: ${batch.modelViewProjMatrix}")

										val pl = getProgramLayout(batch.state)
										ag.draw(
											type = batch.primType.toAg(),
											vertices = vertexBuffer,
											indices = indexBuffer,
											program = pl.program,
											vertexLayout = pl.layout,
											vertexCount = batch.vertexCount,
											uniforms = mapOf(
												u_modelViewProjMatrix to batch.modelViewProjMatrix
											)
										)
									}
								}
							}
						}
					}
					mem.write(display.address, tempBmp.data)
				} finally {
					batchesQueue.clear()
				}
			}
		}
	}

	suspend override fun sceneInit(sceneView: Container) {
		val samplesFolder = when {
			OS.isJs -> applicationVfs
		//else -> ResourcesVfs
			else -> applicationVfs["samples"].jail()
		}

		//val elfBytes = samplesFolder["minifire.elf"].readAll()
		//val elfBytes = samplesFolder["HelloWorldPSP.elf"].readAll()
		//val elfBytes = samplesFolder["rtctest.elf"].readAll()
		//val elfBytes = samplesFolder["compilerPerf.elf"].readAll()
		val elfBytes = samplesFolder["cube.elf"].readAll()

		val renderView = KorgeRenderer(this)

		emulator = Emulator(mem = Memory(), gpuRenderer = renderView).apply {
			registerNativeModules()
			loadElfAndSetRegisters(elfBytes.openSync())
			//threadManager.trace("_start")
			//threadManager.trace("user_main")
		}
		val tex by lazy { views.texture(display.bmp) }

		var running = true

		sceneView.addUpdatable {
			if (running) {
				try {
					emulator.frameStep()
				} catch (e: Throwable) {
					e.printStackTrace()
					running = false
				}

				if (display.rawDisplay) {
					display.decodeToBitmap32(display.bmp)
					tex.update(display.bmp)
				}
			}


		}

		sceneView += renderView
		sceneView += views.image(tex)
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