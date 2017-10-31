package com.soywiz.kpspemu

import com.soywiz.korag.AG
import com.soywiz.korag.shader.*
import com.soywiz.korge.Korge
import com.soywiz.korge.input.onKeyDown
import com.soywiz.korge.input.onKeyUp
import com.soywiz.korge.render.RenderContext
import com.soywiz.korge.scene.Module
import com.soywiz.korge.scene.Scene
import com.soywiz.korge.view.*
import com.soywiz.korim.bitmap.Bitmap32
import com.soywiz.korio.JvmStatic
import com.soywiz.korio.inject.AsyncInjector
import com.soywiz.korio.lang.printStackTrace
import com.soywiz.korio.stream.openSync
import com.soywiz.korio.util.OS
import com.soywiz.korio.vfs.applicationVfs
import com.soywiz.korma.Matrix2d
import com.soywiz.korma.geom.SizeInt
import com.soywiz.kpspemu.ctrl.PspCtrlButtons
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
	override val clearEachFrame: Boolean = false
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
						SET(v_Col, a_Col[if (OS.isJs) "bgra" else "rgba"])
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

		private var indexBuffer: AG.Buffer? = null
		private var vertexBuffer: AG.Buffer? = null

		override fun render(ctx: RenderContext, m: Matrix2d) {
			ctx.flush()
			if (batchesQueue.isNotEmpty()) {
				try {
					val ag = ctx.ag
					//mem.read(display.address, tempBmp.data)

					mem.read(display.address, tempBmp.data)

					ag.renderToBitmap(tempBmp) {
						//ag.drawBmp2(Bitmap32(512, 272, premult = true) { x, y -> RGBA.pack(0, y / 2, x / 2, 0xFF) })
						ag.drawBmp(tempBmp)
						//ag.clear(Colors.BLUE) // @TODO: Remove this

						for (batches in batchesQueue) {
							for (batch in batches) {
								if (indexBuffer == null) indexBuffer = ag.createIndexBuffer()
								if (vertexBuffer == null) vertexBuffer = ag.createVertexBuffer()

								indexBuffer!!.upload(batch.indices)
								vertexBuffer!!.upload(batch.vertices)

								//println("----------------")
								//println("indices: ${batch.indices.toList()}")
								//println("primitive: ${batch.primType.toAg()}")
								//println("vertexCount: ${batch.vertexCount}")
								//println("vertexType: ${batch.state.vertexType.hex}")
								//println("vertices: ${batch.vertices.hex}")
								//println("matrix: ${batch.modelViewProjMatrix}")
								//val vr = VertexReader()
								//println(vr.read(batch.vtype, batch.vertices.size / batch.vtype.size(), batch.vertices.openSync()))

								val pl = getProgramLayout(batch.state)
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
									blending = AG.Blending.NONE
								)
							}
						}
					}
					mem.write(display.address, tempBmp.data)
				} finally {
					batchesQueue.clear()
				}
			}

			//ctx.batch.drawQuad(scene.tex, m = m, blendFactors = AG.Blending.NONE)
			ctx.ag.drawBmp(display.bmp)
		}
	}

	val tex by lazy { views.texture(display.bmp) }

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
		//val elfBytes = samplesFolder["cube.elf"].readAll()
		val elfBytes = samplesFolder["ortho.elf"].readAll()

		val renderView = KorgeRenderer(this)

		emulator = Emulator(mem = Memory(), gpuRenderer = renderView).apply {
			registerNativeModules()
			loadElfAndSetRegisters(elfBytes.openSync())
			//threadManager.trace("_start")
			//threadManager.trace("user_main")
		}

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
					//tex.update(display.bmp)
				}
			}
		}

		val keys = BooleanArray(256)

		fun updateKey(keyCode: Int, pressed: Boolean) {
			println("updateKey: $keyCode, $pressed")
			keys[keyCode and 0xFF] = pressed
			when (keyCode) {
				10 -> controller.updateButton(PspCtrlButtons.start, pressed) // return
				32 -> controller.updateButton(PspCtrlButtons.select, pressed) // space
				87 -> controller.updateButton(PspCtrlButtons.triangle, pressed) // W
				65 -> controller.updateButton(PspCtrlButtons.square, pressed) // A
				83 -> controller.updateButton(PspCtrlButtons.cross, pressed) // S
				68 -> controller.updateButton(PspCtrlButtons.circle, pressed) // D
				81 -> controller.updateButton(PspCtrlButtons.leftTrigger, pressed) // Q
				69 -> controller.updateButton(PspCtrlButtons.rightTrigger, pressed) // E
				37 -> controller.updateButton(PspCtrlButtons.left, pressed) // LEFT
				38 -> controller.updateButton(PspCtrlButtons.up, pressed) // UP
				39 -> controller.updateButton(PspCtrlButtons.right, pressed) // RIGHT
				40 -> controller.updateButton(PspCtrlButtons.down, pressed) // DOWN
				in 73..76 -> Unit // IJKL (analog)
				else -> println("Unhandled($pressed): $keyCode")
			}

			controller.updateAnalog(
				x = when { keys[74] -> -1f; keys[76] -> +1f; else -> 0f; },
				y = when { keys[73] -> +1f; keys[75] -> -1f; else -> 0f; }
			)
		}

		sceneView.onKeyDown { updateKey(it.keyCode, true) }
		sceneView.onKeyUp { updateKey(it.keyCode, false) }

		sceneView += renderView
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
