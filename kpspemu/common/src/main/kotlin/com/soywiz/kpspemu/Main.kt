package com.soywiz.kpspemu

import com.soywiz.korge.Korge
import com.soywiz.korge.scene.Module
import com.soywiz.korge.scene.Scene
import com.soywiz.korge.view.Container
import com.soywiz.korio.JvmStatic
import com.soywiz.korma.geom.SizeInt
import kotlin.reflect.KClass

object Main {
	@JvmStatic fun main(args: Array<String>) = Korge(KpspemuModule)
}

object KpspemuModule : Module() {
	override val mainScene: KClass<out Scene> = KpspemuMainScene::class
	override val size: SizeInt get() = SizeInt(480, 272)
}

class KpspemuMainScene : Scene() {
	suspend override fun sceneInit(sceneView: Container) {
	}
}