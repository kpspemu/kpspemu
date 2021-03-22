package com.soywiz.kpspemu.ui

import com.soywiz.korge.input.*
import com.soywiz.korge.scene.*
import com.soywiz.korge.view.*
import com.soywiz.korim.color.*
import com.soywiz.korim.font.*

fun Views.simpleButton(text: String, width: Int = 80, height: Int = 18, font: BitmapFont = debugBmpFont): View {
    val colorOver = RGBA(0xA0, 0xA0, 0xA0, 0xFF)
    val colorOut = RGBA(0x90, 0x90, 0x90, 0xFF)

    return Container().apply {
        val bg = solidRect(width, height, colorOut)
        text(text, font = font, textSize = height.toDouble() - 2.0) {
            this.name = "label"
            this.x = 4.0
            this.y = 2.0
            //this.bgcolor = Colors.GREEN
            this.autoSize = true
            //this.autoSize = false
            //this.textBounds.setBounds(0, 0, width - 8, height - 8)
            //this.width = width - 8.0
            //this.height = height - 8.0
            //this.enabled = false
            //this.mouseEnabled = false
            //this.enabled = false
        }
        onOut { bg.colorMul = colorOut }
        onOver { bg.colorMul = colorOver }
    }
}
