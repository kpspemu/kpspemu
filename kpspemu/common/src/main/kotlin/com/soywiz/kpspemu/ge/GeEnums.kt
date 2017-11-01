package com.soywiz.kpspemu.ge

import com.soywiz.kpspemu.util.IdEnum

enum class CullingDirection(override val id: Int) : IdEnum {
	CounterClockWise(0),
	ClockWise(1);
	companion object : IdEnum.SmallCompanion<CullingDirection>(values())
}

enum class SyncType(override val id: Int) : IdEnum {
	WaitForCompletion(0),
	Peek(1);

	companion object : IdEnum.SmallCompanion<SyncType>(values())
}

enum class DisplayListStatus(override val id: Int) : IdEnum {
	Completed(0), // The list has been completed (PSP_GE_LIST_COMPLETED)
	Queued(1), // list is queued but not executed yet (PSP_GE_LIST_QUEUED)
	Drawing(2), // The list is currently being executed (PSP_GE_LIST_DRAWING)
	Stalling(3), // The list was stopped because it encountered stall address (PSP_GE_LIST_STALLING)
	Paused(4); // The list is paused because of a signal or sceGeBreak (PSP_GE_LIST_PAUSED)

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
	Void(0, nbytes = 0),
	Invalid1(1, nbytes = 0),
	Invalid2(2, nbytes = 0),
	Invalid3(3, nbytes = 0),
	Color5650(4, nbytes = 2),
	Color5551(5, nbytes = 2),
	Color4444(6, nbytes = 2),
	Color8888(7, nbytes = 4);

	companion object : IdEnum.SmallCompanion<ColorEnum>(values())
}

enum class LightTypeEnum(override val id: Int) : IdEnum {
	Directional(0), PointLight(1), SpotLight(2);

	companion object : IdEnum.SmallCompanion<LightTypeEnum>(values())
}

enum class LightModelEnum(override val id: Int) : IdEnum {
	SingleColor(0), SeparateSpecularColor(1);

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
	Auto(0), Const(1), Slope(2);

	companion object : IdEnum.SmallCompanion<TextureLevelMode>(values())
}

enum class TestFunctionEnum(override val id: Int) : IdEnum {
	Never(0),
	Always(1),
	Equal(2),
	NotEqual(3),
	Less(4),
	LessOrEqual(5),
	Greater(6),
	GreaterOrEqual(7);

	companion object : IdEnum.SmallCompanion<TestFunctionEnum>(values())
}

enum class ShadingModelEnum(override val id: Int) : IdEnum {
	Flat(0),
	Smooth(1);

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
	Add(0),
	Substract(1),
	ReverseSubstract(2),
	Min(3),
	Max(4),
	Abs(5);

	companion object : IdEnum.SmallCompanion<GuBlendingEquation>(values())
}

enum class StencilOperationEnum(override val id: Int) : IdEnum {
	Keep(0),
	Zero(1),
	Replace(2),
	Invert(3),
	Increment(4),
	Decrement(5);

	companion object : IdEnum.SmallCompanion<StencilOperationEnum>(values())
}

enum class WrapMode(override val id: Int) : IdEnum {
	Repeat(0),
	Clamp(1);

	companion object : IdEnum.SmallCompanion<WrapMode>(values())
}

enum class TextureEffect(override val id: Int) : IdEnum {
	Modulate(0),  // GU_TFX_MODULATE
	Decal(1),     // GU_TFX_DECAL
	Blend(2),     // GU_TFX_BLEND
	Replace(3),   // GU_TFX_REPLACE
	Add(4);       // GU_TFX_ADD

	companion object : IdEnum.SmallCompanion<TextureEffect>(values())
}

enum class TextureFilter(override val id: Int) : IdEnum {
	Nearest(0),
	Linear(1),
	NearestMipmapNearest(4),
	LinearMipmapNearest(5),
	NearestMipmapLinear(6),
	LinearMipmapLinear(7);

	companion object : IdEnum.SmallCompanion<TextureFilter>(values())
}

enum class TextureColorComponent(override val id: Int) : IdEnum {
	Rgb(0),    // GU_TCC_RGB
	Rgba(1);   // GU_TCC_RGBA

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