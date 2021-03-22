package com.soywiz.kpspemu.ge

import com.soywiz.korim.color.*
import com.soywiz.korio.util.*

enum class CullingDirection(override val id: Int) : IdEnum {
    COUNTER_CLOCK_WISE(0),
    CLOCK_WISE(1);

    companion object : IdEnum.SmallCompanion<CullingDirection>(values())
}

enum class SyncType(override val id: Int) : IdEnum {
    WAIT_FOR_COMPLETION(0),
    PEEK(1);

    companion object : IdEnum.SmallCompanion<SyncType>(values())
}

enum class DisplayListStatus(override val id: Int) : IdEnum {
    COMPLETED(0), // The list has been completed (PSP_GE_LIST_COMPLETED)
    QUEUED(1), // list is queued but not executed yet (PSP_GE_LIST_QUEUED)
    DRAWING(2), // The list is currently being executed (PSP_GE_LIST_DRAWING)
    STALLING(3), // The list was stopped because it encountered stall address (PSP_GE_LIST_STALLING)
    PAUSED(4); // The list is paused because of a signal or sceGeBreak (PSP_GE_LIST_PAUSED)

    companion object : IdEnum.SmallCompanion<DisplayListStatus>(values())
}

enum class IndexEnum(override val id: Int, val nbytes: Int) : IdEnum {
    VOID(0, nbytes = 0),
    BYTE(1, nbytes = 1),
    SHORT(2, nbytes = 2);

    companion object : IdEnum.SmallCompanion<IndexEnum>(values())
}

enum class NumericEnum(override val id: Int, val nbytes: Int) : IdEnum {
    VOID(0, nbytes = 0),
    BYTE(1, nbytes = 1),
    SHORT(2, nbytes = 2),
    FLOAT(3, nbytes = 4);

    companion object : IdEnum.SmallCompanion<NumericEnum>(values())
}

enum class ColorEnum(override val id: Int, val nbytes: Int) : IdEnum {
    VOID(0, nbytes = 0),
    INVALID1(1, nbytes = 0),
    INVALID2(2, nbytes = 0),
    INVALID3(3, nbytes = 0),
    COLOR5650(4, nbytes = 2),
    COLOR5551(5, nbytes = 2),
    COLOR4444(6, nbytes = 2),
    COLOR8888(7, nbytes = 4);

    companion object : IdEnum.SmallCompanion<ColorEnum>(values())
}

enum class LightTypeEnum(override val id: Int) : IdEnum {
    DIRECTIONAL(0),
    POINT_LIGHT(1),
    SPOT_LIGHT(2);

    companion object : IdEnum.SmallCompanion<LightTypeEnum>(values())
}

enum class LightModelEnum(override val id: Int) : IdEnum {
    SINGLE_COLOR(0),
    SEPARATE_SPECULAR_COLOR(1);

    companion object : IdEnum.SmallCompanion<LightModelEnum>(values())
}

enum class TextureProjectionMapMode(override val id: Int) : IdEnum {
    GU_POSITION(0), // TMAP_TEXTURE_PROJECTION_MODE_POSITION - 3 texture components
    GU_UV(1), // TMAP_TEXTURE_PROJECTION_MODE_TEXTURE_COORDINATES - 2 texture components
    GU_NORMALIZED_NORMAL(2), // TMAP_TEXTURE_PROJECTION_MODE_NORMALIZED_NORMAL - 3 texture components
    GU_NORMAL(3); // TMAP_TEXTURE_PROJECTION_MODE_NORMAL - 3 texture components

    companion object : IdEnum.SmallCompanion<TextureProjectionMapMode>(values())
}

enum class TextureMapMode(override val id: Int) : IdEnum {
    GU_TEXTURE_COORDS(0),
    GU_TEXTURE_MATRIX(1),
    GU_ENVIRONMENT_MAP(2);

    companion object : IdEnum.SmallCompanion<TextureMapMode>(values())
}

enum class TextureLevelMode(override val id: Int) : IdEnum {
    AUTO(0),
    CONST(1),
    SLOPE(2);

    companion object : IdEnum.SmallCompanion<TextureLevelMode>(values())
}

enum class TestFunctionEnum(override val id: Int) : IdEnum {
    NEVER(0),
    ALWAYS(1),
    EQUAL(2),
    NOT_EQUAL(3),
    LESS(4),
    LESS_OR_EQUAL(5),
    GREATER(6),
    GREATER_OR_EQUAL(7);

    companion object : IdEnum.SmallCompanion<TestFunctionEnum>(values())
}

enum class ShadingModelEnum(override val id: Int) : IdEnum {
    FLAT(0),
    SMOOTH(1);

    companion object : IdEnum.SmallCompanion<ShadingModelEnum>(values())
}

enum class GuBlendingFactor(override val id: Int) : IdEnum {
    GU_SRC_COLOR(0),// = 0x0300,
    GU_ONE_MINUS_SRC_COLOR(1),// = 0x0301,
    GU_SRC_ALPHA(2),// = 0x0302,
    GU_ONE_MINUS_SRC_ALPHA(3),// = 0x0303,
    GU_DST_ALPHA(4),// = 0x0304,
    GU_ONE_MINUS_DST_ALPHA(5),// = 0x0305,
    GU_FIX(10);

    companion object : IdEnum.SmallCompanion<GuBlendingFactor>(values())
}

enum class GuBlendingEquation(override val id: Int) : IdEnum {
    ADD(0),
    SUBSTRACT(1),
    REVERSE_SUBSTRACT(2),
    MIN(3),
    MAX(4),
    ABS(5);

    companion object : IdEnum.SmallCompanion<GuBlendingEquation>(values())
}

enum class StencilOperationEnum(override val id: Int) : IdEnum {
    KEEP(0),
    ZERO(1),
    REPLACE(2),
    INVERT(3),
    INCREMENT(4),
    DECREMENT(5);

    companion object : IdEnum.SmallCompanion<StencilOperationEnum>(values())
}

enum class WrapMode(override val id: Int) : IdEnum {
    REPEAT(0),
    CLAMP(1);

    companion object : IdEnum.SmallCompanion<WrapMode>(values())
}

enum class TextureEffect(override val id: Int) : IdEnum {
    MODULATE(0),  // GU_TFX_MODULATE
    DECAL(1),     // GU_TFX_DECAL
    BLEND(2),     // GU_TFX_BLEND
    REPLACE(3),   // GU_TFX_REPLACE
    ADD(4);       // GU_TFX_ADD

    companion object : IdEnum.SmallCompanion<TextureEffect>(values())
}

enum class TextureFilter(override val id: Int, val nearest: Boolean, val nearestMipmap: Boolean) : IdEnum {
    NEAREST(0, nearest = true, nearestMipmap = false),
    LINEAR(1, nearest = false, nearestMipmap = false),
    NEAREST_MIPMAP_NEAREST(4, nearest = true, nearestMipmap = true),
    LINEAR_MIPMAP_NEAREST(5, nearest = false, nearestMipmap = true),
    NEAREST_MIPMAP_LINEAR(6, nearest = true, nearestMipmap = false),
    LINEAR_MIPMAP_LINEAR(7, nearest = false, nearestMipmap = false);

    companion object : IdEnum.SmallCompanion<TextureFilter>(values())
}

enum class TextureColorComponent(override val id: Int) : IdEnum {
    RGB(0),    // GU_TCC_RGB
    RGBA(1);   // GU_TCC_RGBA

    companion object : IdEnum.SmallCompanion<TextureColorComponent>(values())
}

enum class PrimitiveType(override val id: Int) : IdEnum {
    POINTS(0),
    LINES(1),
    LINE_STRIP(2),
    TRIANGLES(3),
    TRIANGLE_STRIP(4),
    TRIANGLE_FAN(5),
    SPRITES(6);

    companion object : IdEnum.SmallCompanion<PrimitiveType>(values())
}

object ClearBufferSet {
    val ColorBuffer = 1
    val StencilBuffer = 2
    val DepthBuffer = 4
    val FastClear = 16
}

object PspRGB_565 : ColorFormat16, ColorFormat by ColorFormat.Mixin(
    bpp = 16,
    rOffset = 0, rSize = 5,
    gOffset = 5, gSize = 6,
    bOffset = 11, bSize = 5,
    aOffset = 15, aSize = 0
) {
    override fun getA(v: Int): Int = 0xFF
}

object PspRGBA_5551 : ColorFormat16, ColorFormat by ColorFormat.Mixin(
    bpp = 16,
    rOffset = 0, rSize = 5,
    gOffset = 5, gSize = 5,
    bOffset = 10, bSize = 5,
    aOffset = 15, aSize = 1
) {
    // Reverse alpha bit!
    //override fun getA(v: Int): Int = if (v.extractScaledFFDefault(15, 1, default = 0xFF) == 0) 0xFF else 0x00
}

// @TODO: kotlin-native bug: https://github.com/JetBrains/kotlin-native/issues/1779
/*
enum class PixelFormat(
    override val id: Int,
    val bytesPerPixel: Double,
    val colorFormat: ColorFormat? = null,
    val isRgba: Boolean = false,
    val isPalette: Boolean = false,
    val colorBits: Int = 0,
    val paletteBits: Int = 0,
    val dxtVersion: Int = 0,
    val isCompressed: Boolean = false
) : IdEnum {
    RGBA_5650(0, bytesPerPixel = 2.0, colorFormat = PspRGB_565, isRgba = true, colorBits = 16),
    RGBA_5551(1, bytesPerPixel = 2.0, colorFormat = PspRGBA_5551, isRgba = true, colorBits = 16),
    RGBA_4444(2, bytesPerPixel = 2.0, colorFormat = com.soywiz.korim.color.RGBA_4444, isRgba = true, colorBits = 16),
    RGBA_8888(3, bytesPerPixel = 4.0, colorFormat = com.soywiz.korim.color.RGBA, isRgba = true, colorBits = 32),
    PALETTE_T4(4, bytesPerPixel = 0.5, isPalette = true, paletteBits = 4),
    PALETTE_T8(5, bytesPerPixel = 1.0, isPalette = true, paletteBits = 8),
    PALETTE_T16(6, bytesPerPixel = 2.0, isPalette = true, paletteBits = 16),
    PALETTE_T32(7, bytesPerPixel = 4.0, isPalette = true, paletteBits = 32),
    COMPRESSED_DXT1(8, bytesPerPixel = 0.5, isCompressed = true, dxtVersion = 1),
    COMPRESSED_DXT3(9, bytesPerPixel = 1.0, isCompressed = true, dxtVersion = 3),
    COMPRESSED_DXT5(10, bytesPerPixel = 1.0, isCompressed = true, dxtVersion = 5);

    val bitsPerPixel = (bytesPerPixel * 8).toInt()

    val requireClut: Boolean = isPalette
    fun getSizeInBytes(count: Int): Int = (bytesPerPixel * count).toInt()

    companion object : IdEnum.SmallCompanion<PixelFormat>(values())
}
*/

enum class PixelFormat(
    override val id: Int,
    val bytesPerPixel: Double,
    val isRgba: Boolean = false,
    val isPalette: Boolean = false,
    val colorBits: Int = 0,
    val paletteBits: Int = 0,
    val dxtVersion: Int = 0,
    val isCompressed: Boolean = false
) : IdEnum {
    RGBA_5650(0, 2.0, true, false, 16, 0, 0, false),
    RGBA_5551(1, 2.0, true, false, 16, 0, 0, false),
    RGBA_4444(2, 2.0, true, false, 16, 0, 0, false),
    RGBA_8888(3, 4.0, true, false, 32, 0, 0, false),
    PALETTE_T4(4, 0.5, false,  true, 0, 4, 0, false),
    PALETTE_T8(5, 1.0, false,  true, 0, 8, 0, false),
    PALETTE_T16(6, 2.0, false, true, 0, 16, 0, false),
    PALETTE_T32(7, 4.0, false, true, 0, 32, 0, false),
    COMPRESSED_DXT1(8, 0.5, false,  false, 0, 0, 1, true),
    COMPRESSED_DXT3(9, 1.0, false,  false, 0, 0, 3, true),
    COMPRESSED_DXT5(10, 1.0, false, false, 0, 0, 5, true);

    val bitsPerPixel get() = (bytesPerPixel * 8).toInt()

    val requireClut: Boolean get() = isPalette
    fun getSizeInBytes(count: Int): Int = (bytesPerPixel * count).toInt()

    val colorFormat: ColorFormat?
        get() = when (this) {
            RGBA_5650 -> PspRGB_565
            RGBA_5551 -> PspRGBA_5551
            RGBA_4444 -> com.soywiz.korim.color.RGBA_4444
            RGBA_8888 -> com.soywiz.korim.color.RGBA
            else -> null
        }

    companion object : IdEnum.SmallCompanion<PixelFormat>(values())
}
