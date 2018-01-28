package com.soywiz.kpspemu.ui

import com.soywiz.korge.bitmapfont.BitmapFont
import com.soywiz.korge.input.onOut
import com.soywiz.korge.input.onOver
import com.soywiz.korge.view.View
import com.soywiz.korge.view.Views
import com.soywiz.korge.view.text
import com.soywiz.korim.color.RGBA

fun Views.simpleButton(text: String, width: Int = 80, height: Int = 18, font: BitmapFont = this.defaultFont): View {
    val button = container()
    val colorOver = RGBA(0xA0, 0xA0, 0xA0, 0xFF)
    val colorOut = RGBA(0x90, 0x90, 0x90, 0xFF)

    val bg = solidRect(width, height, colorOut)
    val txt = text(text, font = font, textSize = height.toDouble() - 2.0).apply {
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
    //txt.format = Html.Format(face = Html.FontFace.Bitmap(font), size = height)
    //txt.text = text
    button += bg
    button += txt
    button.onOut { bg.colorMul = colorOut }
    button.onOver { bg.colorMul = colorOver }
    //txt.textBounds.setBounds(0, 0, 50, 50)
    //println("---------------")
    //println(txt.textBounds)
    //println(txt.globalBounds)
    return button
}
