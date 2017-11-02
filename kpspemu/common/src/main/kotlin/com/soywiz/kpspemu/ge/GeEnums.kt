package com.soywiz.kpspemu.ge

import com.soywiz.kpspemu.util.IdEnum

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

enum class TextureFilter(override val id: Int) : IdEnum {
	NEAREST(0),
	LINEAR(1),
	NEAREST_MIPMAP_NEAREST(4),
	LINEAR_MIPMAP_NEAREST(5),
	NEAREST_MIPMAP_LINEAR(6),
	LINEAR_MIPMAP_LINEAR(7);

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