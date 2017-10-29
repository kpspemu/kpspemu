package com.soywiz.kpspemu.hle

import com.soywiz.kpspemu.util.NumericEnum

enum class PixelFormat(override val id: Int) : NumericEnum {
	RGBA_5650(0),
	RGBA_5551(1),
	RGBA_4444(2),
	RGBA_8888(3),
	PALETTE_T4(4),
	PALETTE_T8(5),
	PALETTE_T16(6),
	PALETTE_T32(7),
	COMPRESSED_DXT1(8),
	COMPRESSED_DXT3(9),
	COMPRESSED_DXT5(10);

	companion object {
		val values = values()
		operator fun invoke(id: Int) = values[id]
	}
}
