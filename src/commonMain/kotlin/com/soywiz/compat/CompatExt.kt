package com.soywiz.compat

import com.soywiz.korge.view.*
import com.soywiz.korim.atlas.*
import com.soywiz.korim.bitmap.*
import com.soywiz.korim.format.*
import com.soywiz.korio.dynamic.*
import com.soywiz.korio.file.*
import com.soywiz.korio.serialization.json.*
import com.soywiz.korma.geom.*

var View.enabled: Boolean
    get() = false
    set(value) { }

// @TODO: Move to KorGE
suspend fun VfsFile.readAtlas2(): Atlas {
    val atlas = this.readString().fromJson().dyn
    val width = atlas["width"].int
    val height = atlas["height"].int
    val fileName = atlas["file"].str
    val bitmap = this.parent[fileName].readBitmapSlice()
    val meta = AtlasInfo.Meta(image = fileName, size = AtlasInfo.Size(width, height))
    return Atlas(bitmap, AtlasInfo(meta = meta, pages = listOf(AtlasInfo.Page(
        fileName = fileName,
        size = AtlasInfo.Size(width, height),
        format = "png",
        filterMin = true,
        filterMag = true,
        repeatX = false,
        repeatY = false,
        regions = atlas["sprites"].list.map {
            val rotated = it["rotated"].bool
            val extruded = it["extruded"].bool
            val x = it["x"].int
            val y = it["y"].int
            val w = it["w"].int
            val h = it["h"].int
            val margin = it["margin"].int
            val name = it["name"].str
            AtlasInfo.Region(name, AtlasInfo.Rect(x, y, w, h), rotated, AtlasInfo.Size(w, h), AtlasInfo.Rect(x, y, w, h), false, orig = AtlasInfo.Size(0, 0), offset = Point(0, 0))
        }
    ))))

    //return Atlas(atlas["sprites"].list.map {
//
    //    bitmap.sliceWithSize(x, y, w, h, name).withName(name)
    //}).also {
    //    println("atlas: $it")
    //}
}