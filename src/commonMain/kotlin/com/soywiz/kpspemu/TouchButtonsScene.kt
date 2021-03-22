package com.soywiz.kpspemu

import com.soywiz.compat.*
import com.soywiz.korev.*
import com.soywiz.korge.component.*
import com.soywiz.korge.input.*
import com.soywiz.korge.scene.*
import com.soywiz.korge.view.*
import com.soywiz.korim.atlas.*
import com.soywiz.korio.async.*
import com.soywiz.korio.file.std.*
import com.soywiz.korma.geom.*
import com.soywiz.kpspemu.ctrl.*
import kotlin.math.*

class TouchButtonsScene(val emulator: Emulator) : Scene() {
    val controller get() = emulator.controller
    lateinit var atlas: Atlas

    override suspend fun Container.sceneInit() {
        val sceneView = this
        //atlas = rootLocalVfs["buttons.json"].readAtlas2(views)
        atlas = resourcesVfs["buttons.json"].readAtlas2()

        sceneView.scale = 480.0 / 1280.0

        val buttonsPos = Point(1100, 720 / 2)
        val dpadPos = Point(170, 300)
        val pseparation = 0.15
        val bseparation = 0.2
        val pscale = 1.1
        val bscale = 1.1

        addButton("up.png", PspCtrlButtons.up, dpadPos, 0.5, 1.0 + pseparation, pscale)
        addButton("left.png", PspCtrlButtons.left, dpadPos, 1.0 + pseparation, 0.5, pscale)
        addButton("right.png", PspCtrlButtons.right, dpadPos, -pseparation, 0.5, pscale)
        addButton("down.png", PspCtrlButtons.down, dpadPos, 0.5, -pseparation, pscale)

        addButton("triangle.png", PspCtrlButtons.triangle, buttonsPos, 0.5, 1.0 + bseparation, bscale)
        addButton("square.png", PspCtrlButtons.square, buttonsPos, 1.0 + bseparation, 0.5, bscale)
        addButton("circle.png", PspCtrlButtons.circle, buttonsPos, -bseparation, 0.5, bscale)
        addButton("cross.png", PspCtrlButtons.cross, buttonsPos, 0.5, -bseparation, bscale)

        addButton("trigger_l.png", PspCtrlButtons.leftTrigger, Point(160, 0), 0.5, 0.0)
        addButton("trigger_r.png", PspCtrlButtons.rightTrigger, Point(1280 - 160, 0), 0.5, 0.0)

        addButton("start.png", PspCtrlButtons.start, Point(1280 - 160, 720), 0.5, 1.0)
        addButton("select.png", PspCtrlButtons.select, Point(1280 - 380, 720), 0.5, 1.0)

        addButton("home.png", PspCtrlButtons.home, Point(1280 / 2, 0), 1.1, 0.0).apply {
            view.onClick {
                println("home.onClick")
                emulator.onHomePress()
            }
        }
        addButton("load.png", PspCtrlButtons.hold, Point(1280 / 2, 0), -0.1, 0.0).apply {
            view.onClick {
                println("load.onClick")
                emulator.onLoadPress()
            }
        }

        addThumb(Point(172.0, 600.0))

        sceneView.addComponent(object : TouchComponent {
            override val view: View = sceneView

            override fun onTouchEvent(views: Views, e: TouchEvent) {
                updateEvent()
            }

            fun View.testAnyTouch(): Boolean {
                for (touch in views.input.activeTouches) {
                    if (touch.id == thumbTouchId) continue // Ignore the thumb touch
                    //if (hitTest(touch.current) != null) return true
                    println("Missing hitTest(touch.current)")
                }
                return false
            }

            fun updateEvent() {
                //println("views.nativeMouseX=${views.nativeMouseX}, views.nativeMouseY=${views.nativeMouseY}")
                for (button in buttons) {
                    if (button.view.testAnyTouch()) {
                        button.view.alpha = alphaDown
                        controller.updateButton(button.button, true)
                    } else {
                        button.view.alpha = alphaUp
                        controller.updateButton(button.button, false)
                    }
                }
            }

        })

        updateTouch()
        sceneView.gamepad {
            connected { updateTouch() }
            disconnected { updateTouch() }
        }
        sceneView.onStageResized { width, height ->
            //println("resized:" + views.input.isTouchDevice)
            updateTouch()
        }
    }

    var View.visibleEnabled: Boolean
        get() = visible
        set(value) {
            visible = value
            mouseEnabled = value
        }

    fun updateTouch() {
        val touch = views.input.connectedGamepads.isEmpty() && views.input.isTouchDevice
        thumbContainer.visibleEnabled = touch

        for (button in buttons) {
            button.view.visibleEnabled = touch
            if (button.button == PspCtrlButtons.home || button.button == PspCtrlButtons.hold) {
                button.view.visibleEnabled = true
            }
        }
    }

    val alphaUp = 0.2
    val alphaDown = 0.5

    class Button(val button: PspCtrlButtons, val view: View) {
        val onClick = Signal<Unit>()
    }

    val buttons = arrayListOf<Button>()

    fun addButton(
        file: String,
        pspButton: PspCtrlButtons,
        pos: Point,
        anchorX: Double,
        anchorY: Double,
        scale: Double = 1.0
    ): Button {
        //onDown { this.alpha = alphaDown; controller.updateButton(button, true) }
        //onDownFromOutside { this.alpha = alphaDown; controller.updateButton(button, true) }
        //onUpAnywhere{ this.alpha = alphaUp; controller.updateButton(button, false) }
        val button = Button(pspButton, Image(atlas[file]).apply {
            this.x = pos.x
            this.y = pos.y
            this.anchorX = anchorX
            this.anchorY = anchorY
            this.alpha = alphaUp
            this.scale = scale
        })
        button.view.alpha = alphaUp
        button.view.onOver { button.view.alpha = alphaDown }
        button.view.onOut { button.view.alpha = alphaUp }
        button.view.onClick { button.onClick(Unit) }
        buttons += button
        sceneView += button.view
        return button
    }

    val thumbTouchIdNone = Int.MIN_VALUE
    var thumbTouchId = thumbTouchIdNone
    lateinit var thumbContainer: Container

    fun addThumb(pos: Point) {
        thumbContainer = Container().apply {
            sceneView += this
            this.x = pos.x
            this.y = pos.y
            val bg = image(atlas["thumb_bg.png"]).apply {
                this.anchorX = 0.5
                this.anchorY = 0.5
                this.alpha = 0.2
            }
            val thumb = image(atlas["thumb.png"]).apply {
                this.anchorX = 0.5
                this.anchorY = 0.5
                this.alpha = 0.2
            }
            bg.apply {
                launchImmediately {
                    onMouseDrag {
                        if (it.start){
                            //thumbTouchId = it.id
                            thumbTouchId = 0
                            //println("START")
                            bg.alpha = alphaDown
                            thumb.alpha = alphaDown
                        } else if (it.end) {
                            thumbTouchId = thumbTouchIdNone
                            //println("END")
                            thumb.x = 0.0
                            thumb.y = 0.0
                            bg.alpha = alphaUp
                            thumb.alpha = alphaUp
                            controller.updateAnalog(0f, 0f)
                        } else {
                            //println("Moving: $it")
                            it.dx
                            val angle = atan2(it.dx, it.dy)

                            val magnitude = min(32.0, hypot(it.dx, it.dy))

                            thumb.x = sin(angle) * magnitude
                            thumb.y = cos(angle) * magnitude
                            controller.updateAnalog(sin(angle).toFloat(), cos(angle).toFloat())
                        }
                    }
                }
            }
        }
    }
}
