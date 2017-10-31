package com.soywiz.kpspemu

import com.soywiz.korag.AG
import com.soywiz.korag.geom.Matrix4
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
import com.soywiz.korim.color.Colors
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
import com.soywiz.kpspemu.ge.GeBatch
import com.soywiz.kpspemu.ge.GpuRenderer
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

		//val a_Tex = Attribute("a_Tex", VarType.Float2, normalized = false)
		val a_Tex = Attribute("a_Tex", VarType.Float2, normalized = false)
		val a_Col = Attribute("a_Col", VarType.Byte4, normalized = false)
		val a_Pos = Attribute("a_Pos", VarType.Float3, normalized = false)

		val layout = VertexLayout(a_Tex.inactived(), a_Col.inactived(), a_Pos)

		val program = Program(
			vertex = VertexShader {
				//SET(out, vec4(a_Pos, 0f.lit, 1f.lit) * u_ProjMat)
				SET(out, u_modelViewProjMatrix * vec4(a_Pos, 1f.lit))
				//SET(out, vec4(a_Pos, 1f.lit))
			},
			fragment = FragmentShader {
				out set vec4(1f.lit, 0f.lit, 0f.lit, 1f.lit)
			},
			name = "PROGRAM_DEBUG"
		)

		//data class DrawState(val program: Program, )

		val modelViewProjMatrix = Matrix4()
		val projMatrix = Matrix4().setToIdentity()
		val viewMatrix = Matrix4().setToIdentity()
		val worldMatrix = Matrix4().setToIdentity()
		val uniforms = mapOf(u_modelViewProjMatrix to modelViewProjMatrix)

		override fun render(ctx: RenderContext, m: Matrix2d) {
			if (batchesQueue.isNotEmpty()) {
				try {
					val ag = ctx.ag
					ag.renderToBitmap(tempBmp) {
						ag.clear(Colors.BLUE) // @TODO: Remove this

						for (batches in batchesQueue) {
							for (batch in batches) {
								val ib = ag.createIndexBuffer().use { ib ->
									ag.createVertexBuffer().use { vb ->
										ib.upload(batch.indices)
										vb.upload(batch.vertices)

										if (batch.vertexCount > 10) {
											batch.state.getProjMatrix(projMatrix)
											batch.state.getViewMatrix(viewMatrix)
											batch.state.getWorldMatrix(worldMatrix)

											modelViewProjMatrix.setToIdentity()
											modelViewProjMatrix.setToMultiply(modelViewProjMatrix, projMatrix)
											modelViewProjMatrix.setToMultiply(modelViewProjMatrix, viewMatrix)
											modelViewProjMatrix.setToMultiply(modelViewProjMatrix, worldMatrix)

											ag.draw(vb, program, AG.DrawType.TRIANGLES, layout, batch.vertexCount, indices = ib, uniforms = uniforms)
										}
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