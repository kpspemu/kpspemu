package com.soywiz.kpspemu.hle

import com.soywiz.kpspemu.util.IdEnum

enum class PixelFormat(override val id: Int, val hasClut: Boolean, val bytesPerPixel: Double) : IdEnum {
	RGBA_5650(0, hasClut = false, bytesPerPixel = 2.0),
	RGBA_5551(1, hasClut = false, bytesPerPixel = 2.0),
	RGBA_4444(2, hasClut = false, bytesPerPixel = 2.0),
	RGBA_8888(3, hasClut = false, bytesPerPixel = 4.0),
	PALETTE_T4(4, hasClut = true, bytesPerPixel = 0.5),
	PALETTE_T8(5, hasClut = true, bytesPerPixel = 1.0),
	PALETTE_T16(6, hasClut = true, bytesPerPixel = 2.0),
	PALETTE_T32(7, hasClut = true, bytesPerPixel = 4.0),
	COMPRESSED_DXT1(8, hasClut = false, bytesPerPixel = 0.5),
	COMPRESSED_DXT3(9, hasClut = false, bytesPerPixel = 1.0),
	COMPRESSED_DXT5(10, hasClut = false, bytesPerPixel = 1.0);

	fun getSizeInBytes(count: Int): Int = (bytesPerPixel * count).toInt()

	companion object : IdEnum.SmallCompanion<PixelFormat>(values())
}
