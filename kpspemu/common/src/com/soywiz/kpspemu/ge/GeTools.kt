package com.soywiz.kpspemu.ge

import com.soywiz.kmem.*
import com.soywiz.korim.bitmap.*
import com.soywiz.korim.color.*
import com.soywiz.korio.error.*
import com.soywiz.kpspemu.mem.*
import kotlin.math.*

fun Bitmap32.setTo(
    colorFormat: PixelFormat,
    colorData: ByteArray,
    mem: Memory,
    clutReader: ClutReader? = null,
    swizzled: Boolean = false,
    width: Int = 0,
    height: Int = 0
): Bitmap32 {
    if (swizzled) {
        unswizzleInline(colorFormat, colorData, width, height)
    }
    when {
        colorFormat.requireClut -> {
            //println(clutFormat)
            clutReader!!
            val ncolors = 2.0.pow(colorFormat.paletteBits).toInt()
            val colors = IntArray(ncolors) { clutReader.getColor(mem, it) }


            when (colorFormat.paletteBits) {
                4 -> {
                    //println("PAL4")
                    var m = 0
                    for (n in 0 until this.area / 2) {
                        val byte = colorData[n].toInt() and 0xFF
                        this.data[m++] = RGBA(colors[((byte ushr 0) and 0b1111)])
                        this.data[m++] = RGBA(colors[((byte ushr 4) and 0b1111)])
                    }
                }
                8 -> {
                    var m = 0
                    for (n in 0 until this.area) {
                        val byte = colorData[n]
                        this.data[m++] = RGBA(colors[((byte.toInt() ushr 0) and 0b11111111) % colors.size])
                    }
                }
                else -> {
                    invalidOp("Unsupported palette of size ${colorFormat.paletteBits}")
                }
            }
        }
        colorFormat.isRgba -> {
            val cf = colorFormat.colorFormat!!
            cf.decodeToBitmap32(this, colorData)
        }
        colorFormat.isCompressed -> {
            TODO("Unsupported DXT")
        }
    }
    return this
}

fun unswizzleInline(format: PixelFormat, from: ByteArray, width: Int, height: Int) {
    val rowWidth = format.getSizeInBytes(width)
    val textureHeight = height
    val size = rowWidth * textureHeight
    val temp = ByteArray(size)
    unswizzle(from, temp, rowWidth, textureHeight)
    arraycopy(temp, 0, from, 0, size)
}

private fun unswizzle(input: ByteArray, output: ByteArray, rowWidth: Int, textureHeight: Int) {
    val pitch = ((rowWidth - 16) / 4)
    val bxc = (rowWidth / 16)
    val byc = (textureHeight / 8)
    val pitch4 = (pitch * 4)

    var src = 0
    var ydest = 0
    //for (var by = 0; by < byc; by++) {
    for (by in 0 until byc) {
        var xdest = ydest
        for (bx in 0 until bxc) {
            var dest = xdest
            for (n in 0 until 8) {
                //ArrayBufferUtils.copy(input, src, output, dest, 16);
                arraycopy(input, src, output, dest, 16)
                src += 16
                dest += 16
                dest += pitch4
            }
            xdest += 16
        }
        ydest += rowWidth * 8
    }
}