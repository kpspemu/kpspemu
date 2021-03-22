@file:Suppress("unused", "MemberVisibilityCanPrivate")

package com.soywiz.kpspemu.ge

import com.soywiz.kmem.*
import com.soywiz.korge.util.*
import com.soywiz.korim.color.*
import com.soywiz.korio.error.*
import com.soywiz.korio.stream.*
import com.soywiz.korma.*
import com.soywiz.kpspemu.mem.*
import com.soywiz.kpspemu.util.*
import com.soywiz.krypto.encoding.*


fun bool1(p: Int): Boolean = p != 0
fun parambool(p: Int, offset: Int): Boolean = ((p ushr offset) and 0b1) != 0
fun param1(p: Int, offset: Int): Int = (p ushr offset) and 0b1
fun param2(p: Int, offset: Int): Int = (p ushr offset) and 0b11
fun param3(p: Int, offset: Int): Int = (p ushr offset) and 0b111
fun param4(p: Int, offset: Int): Int = (p ushr offset) and 0b1111
fun param5(p: Int, offset: Int): Int = (p ushr offset) and 0b11111
fun param8(p: Int, offset: Int): Int = (p ushr offset) and 0b11111111
fun param10(p: Int, offset: Int): Int = (p ushr offset) and 0b1111111111
fun param16(p: Int, offset: Int): Int = (p ushr offset) and 0b1111111111111111
fun param24(p: Int): Int = p and 0b111111111111111111111111
fun float1(p: Int) = Float.fromBits(p shl 8)

class VertexState(val data: IntArray) {
    val value get() = param24(this.data[Op.VERTEXTYPE])
    val reversedNormal get() = bool1(this.data[Op.REVERSENORMAL])
    var address: Int
        set(value) = run { this.data[Op.VADDR] = value or (Op.VADDR shl 24) }
        get() = param24(this.data[Op.VADDR])

    val texture get() = NumericEnum(param2(this.data[Op.VERTEXTYPE], 0))
    val color get() = ColorEnum(param3(this.data[Op.VERTEXTYPE], 2))
    val normal get() = NumericEnum(param2(this.data[Op.VERTEXTYPE], 5))
    val position get() = NumericEnum(param2(this.data[Op.VERTEXTYPE], 7))
    val weight get() = NumericEnum(param2(this.data[Op.VERTEXTYPE], 9))
    val index get() = IndexEnum(param2(this.data[Op.VERTEXTYPE], 11))
    val weightCount get() = param3(this.data[Op.VERTEXTYPE], 14)
    val morphingVertexCount get() = param2(this.data[Op.VERTEXTYPE], 18)
    val transform2D get() = parambool(this.data[Op.VERTEXTYPE], 23)
}

//fun createMatrix4x4(data: IntArray, offset: Int) = FloatArray(data.buffer).subarray(offset, offset + 16);
//fun createMatrix4x3(data: Uint32Array, offset: number) = Float32Array (data.buffer).subarray(offset, offset + 12);

class ViewPort(val data: IntArray) {
    val x get() = float1(this.data[Op.VIEWPORTX2])
    val y get() = float1(this.data[Op.VIEWPORTY2])
    val z get() = float1(this.data[Op.VIEWPORTZ2])

    val width get() = float1(this.data[Op.VIEWPORTX1])
    val height get() = float1(this.data[Op.VIEWPORTY1])
    val depth get() = float1(this.data[Op.VIEWPORTZ1])
}

class Region(val data: IntArray) {
    val x1 get() = param10(this.data[Op.REGION1], 0)
    val y1 get() = param10(this.data[Op.REGION1], 10)
    val x2 get() = param10(this.data[Op.REGION2], 0)
    val y2 get() = param10(this.data[Op.REGION2], 10)
}

class Light(val data: IntArray, val index: Int) {
    companion object {
        private val REG_TYPES = intArrayOf(Op.LIGHTTYPE0, Op.LIGHTTYPE1, Op.LIGHTTYPE2, Op.LIGHTTYPE3)
        private val REG_LCA = intArrayOf(Op.LCA0, Op.LCA1, Op.LCA2, Op.LCA3)
        private val REG_LLA = intArrayOf(Op.LLA0, Op.LLA1, Op.LLA2, Op.LLA3)
        private val REG_LQA = intArrayOf(Op.LQA0, Op.LQA1, Op.LQA2, Op.LQA3)
        private val REG_SPOTEXP = intArrayOf(Op.SPOTEXP0, Op.SPOTEXP1, Op.SPOTEXP2, Op.SPOTEXP3)
        private val REG_SPOTCUT = intArrayOf(Op.SPOTCUT0, Op.SPOTCUT1, Op.SPOTCUT2, Op.SPOTCUT3)
        private val LXP = intArrayOf(Op.LXP0, Op.LXP1, Op.LXP2, Op.LXP3)
        private val LYP = intArrayOf(Op.LYP0, Op.LYP1, Op.LYP2, Op.LYP3)
        private val LZP = intArrayOf(Op.LZP0, Op.LZP1, Op.LZP2, Op.LZP3)
        private val LXD = intArrayOf(Op.LXD0, Op.LXD1, Op.LXD2, Op.LXD3)
        private val LYD = intArrayOf(Op.LYD0, Op.LYD1, Op.LYD2, Op.LYD3)
        private val LZD = intArrayOf(Op.LZD0, Op.LZD1, Op.LZD2, Op.LZD3)
        private val ALC = intArrayOf(Op.ALC0, Op.ALC1, Op.ALC2, Op.ALC3)
        private val DLC = intArrayOf(Op.DLC0, Op.DLC1, Op.DLC2, Op.DLC3)
        private val SLC = intArrayOf(Op.SLC0, Op.SLC1, Op.SLC2, Op.SLC3)
    }

    val enabled get() = bool1(this.data[Op.LIGHTENABLE0 + this.index])
    val kind get() = LightModelEnum(param8(this.data[Light.REG_TYPES[this.index]], 0))
    val type get() = LightTypeEnum(param8(this.data[Light.REG_TYPES[this.index]], 8))
    val pw get() = (this.type == LightTypeEnum.SPOT_LIGHT).toInt()
    val px get() = float1(this.data[Light.LXP[this.index]])
    val py get() = float1(this.data[Light.LYP[this.index]])
    val pz get() = float1(this.data[Light.LZP[this.index]])
    val dx get() = float1(this.data[Light.LXD[this.index]])
    val dy get() = float1(this.data[Light.LYD[this.index]])
    val dz get() = float1(this.data[Light.LZD[this.index]])
    val spotExponent get() = float1(this.data[Light.REG_SPOTEXP[this.index]])
    val spotCutoff get() = float1(this.data[Light.REG_SPOTCUT[this.index]])
    val constantAttenuation get() = float1(this.data[Light.REG_LCA[this.index]])
    val linearAttenuation get() = float1(this.data[Light.REG_LLA[this.index]])
    val quadraticAttenuation get() = float1(this.data[Light.REG_LQA[this.index]])
    val ambientColor get() = Color().setRGB(Light.ALC[this.index])
    val diffuseColor get() = Color().setRGB(Light.DLC[this.index])
    val specularColor get() = Color().setRGB(Light.SLC[this.index])
}

class Lightning(val data: IntArray) {
    val lights = listOf(
        Light(data, 0),
        Light(data, 1),
        Light(data, 2),
        Light(data, 3)
    )

    val lightModel get() = LightModelEnum(param8(this.data[Op.LIGHTMODE], 0))
    val specularPower get() = float1(this.data[Op.MATERIALSPECULARCOEF])
    val ambientLightColor get() = Color().setRGB_A(this.data[Op.AMBIENTCOLOR], this.data[Op.AMBIENTALPHA])
    val enabled get() = bool1(this.data[Op.LIGHTINGENABLE])
}

class MipmapState(val texture: TextureState, private val data: IntArray, val index: Int) {
    val bufferWidth get() = param16(this.data[Op.TEXBUFWIDTH0 + this.index], 0)
    val address
        get() = param24(this.data[Op.TEXADDR0 + this.index]) or (param8(
            this.data[Op.TEXBUFWIDTH0 + this.index],
            16
        ) shl 24)
    val addressEnd get() = this.address + this.sizeInBytes
    val textureWidth get() = 1 shl param4(this.data[Op.TSIZE0 + this.index], 0)
    val textureHeight get() = 1 shl param4(this.data[Op.TSIZE0 + this.index], 8)
    val size get() = this.bufferWidth * this.textureHeight
    val sizeInBytes get() = this.texture.pixelFormat.getSizeInBytes(this.size)
}

interface ClutReader {
    val numberOfColors: Int
    fun getRawColor(mem: Memory, n: Int): Int
    fun getColor(mem: Memory, n: Int): Int
}

class ClutState(val data: IntArray) : ClutReader {
    fun getHashFast(): Int =
        (this.data[Op.CMODE] shl 0) + (this.data[Op.CLOAD] shl 8) + (this.data[Op.CLUTADDR] shl 16) + (this.data[Op.CLUTADDRUPPER] shl 24)

    val cmode get() = this.data[Op.CMODE]
    val cload get() = this.data[Op.CLOAD]
    val address get() = param24(this.data[Op.CLUTADDR]) or ((this.data[Op.CLUTADDRUPPER] shl 8) and 0xFF000000.toInt())
    //val addressEnd get() = this.address + this.sizeInBytes
    override val numberOfColors get() = this.data[Op.CLOAD] * 8
    val pixelFormat get() = PixelFormat(param2(this.data[Op.CMODE], 0))
    val colorBits: Int get() = pixelFormat.bitsPerPixel
    val shift get() = param5(this.data[Op.CMODE], 2)
    val mask get() = param8(this.data[Op.CMODE], 8)
    val start get() = param5(this.data[Op.CMODE], 16)
    //val sizeInBytes get() = this.pixelFormat.getSizeInBytes(this.numberOfColors)

    fun getIndex(n: Int) = ((start + n) ushr shift) and mask

    override fun getRawColor(mem: Memory, n: Int): Int = when (colorBits) {
        16 -> mem.lhu(address + getIndex(n) * 2)
        32 -> mem.lw(address + getIndex(n) * 4)
        else -> invalidOp("Invalid palette")
    }

    override fun getColor(mem: Memory, n: Int): Int = pixelFormat.colorFormat!!.unpackToRGBA(getRawColor(mem, n)).rgba
}

class TextureState(val geState: GeState) {
    val data: IntArray = geState.data
    fun getTextureMatrix(out: Matrix4 = Matrix4()) =
        out.apply { getMatrix4x4(this@TextureState.data, Op.MAT_TEXTURE, out) }

    val clut = ClutState(this.data)

    val hasTexture get() = geState.vertex.texture != NumericEnum.VOID
    val hasClut get() = this.pixelFormat.requireClut

    fun getHashSlow(textureData: ByteArray, clutData: ByteArray): String {
        val hash = arrayListOf<Int>()
        hash.add(textureData.contentHashCode())
        hash.add(this.mipmap.address)
        hash.add(this.mipmap.textureWidth)
        hash.add(this.colorComponent.id)
        hash.add(this.mipmap.textureHeight)
        hash.add(this.swizzled.toInt())
        hash.add(this.pixelFormat.id)
        if (this.hasClut) {
            hash.add(this.clut.getHashFast())
            hash.add(clutData.contentHashCode())
        }
        //value += this.clut.getHashFast();
        return hash.joinToString("_")
    }

    val mipmap get() = this.mipmaps[0]

    val mipmaps = listOf(
        MipmapState(this, this.data, 0),
        MipmapState(this, this.data, 1),
        MipmapState(this, this.data, 2),
        MipmapState(this, this.data, 3),
        MipmapState(this, this.data, 4),
        MipmapState(this, this.data, 5),
        MipmapState(this, this.data, 6),
        MipmapState(this, this.data, 7)
    )

    private val envColorColor = Color()

    val wrapU get() = WrapMode(param8(this.data[Op.TWRAP], 0))
    val wrapV get() = WrapMode(param8(this.data[Op.TWRAP], 8))
    val levelMode get() = TextureLevelMode(param8(this.data[Op.TBIAS], 0))
    val mipmapBias get() = param8(this.data[Op.TBIAS], 16) / 16
    val offsetU get() = float1(this.data[Op.TEXOFFSETU])
    val offsetV get() = float1(this.data[Op.TEXOFFSETV])
    val scaleU get() = float1(this.data[Op.TEXSCALEU])
    val scaleV get() = float1(this.data[Op.TEXSCALEV])
    val shadeU get() = param2(this.data[Op.TEXTURE_ENV_MAP_MATRIX], 0)
    val shadeV get() = param2(this.data[Op.TEXTURE_ENV_MAP_MATRIX], 8)
    val effect get() = TextureEffect(param8(this.data[Op.TFUNC], 0))
    val hasAlpha get() = this.colorComponent == TextureColorComponent.RGBA
    val colorComponent get() = TextureColorComponent(param8(this.data[Op.TFUNC], 8))
    val fragment2X get() = param8(this.data[Op.TFUNC], 16) != 0
    val envColor get() = envColorColor.setRGB(param24(this.data[Op.TEC]))
    val pixelFormat get() = PixelFormat(param4(this.data[Op.TPSM], 0))
    val slopeLevel get() = float1(this.data[Op.TSLOPE])
    val swizzled get() = param8(this.data[Op.TMODE], 0) != 0
    val mipmapShareClut get() = param8(this.data[Op.TMODE], 8) != 0
    val mipmapMaxLevel get() = param8(this.data[Op.TMODE], 16) != 0
    val filterMinification get() = TextureFilter(param8(this.data[Op.TFLT], 0))
    val filterMagnification get() = TextureFilter(param8(this.data[Op.TFLT], 8))
    val enabled get() = bool1(this.data[Op.TEXTUREMAPENABLE])
    val textureMapMode get() = TextureMapMode(param8(this.data[Op.TMAP], 0))
    val textureProjectionMapMode get() = TextureProjectionMapMode(param8(this.data[Op.TMAP], 8))
    val tmode get() = this.data[Op.TMODE]

    fun getPixelsSize(size: Int): Int = this.pixelFormat.getSizeInBytes(size)

    val textureComponentsCount: Int
        get() = when (this.textureMapMode) {
            TextureMapMode.GU_TEXTURE_COORDS -> 2
            TextureMapMode.GU_ENVIRONMENT_MAP -> 2
            TextureMapMode.GU_TEXTURE_MATRIX -> when (this.textureProjectionMapMode) {
                TextureProjectionMapMode.GU_NORMAL -> 3
                TextureProjectionMapMode.GU_NORMALIZED_NORMAL -> 3
                TextureProjectionMapMode.GU_POSITION -> 3
                TextureProjectionMapMode.GU_UV -> 2
            }
        }
}

class CullingState(val data: IntArray) {
    val enabled get() = bool1(this.data[Op.CULLFACEENABLE])
    val direction get() = CullingDirection(param24(this.data[Op.CULL]))
}

class DepthTestState(val data: IntArray) {
    val enabled get() = bool1(this.data[Op.ZTESTENABLE])
    val func get() = TestFunctionEnum(param8(this.data[Op.ZTST], 0))
    val mask get() = param16(this.data[Op.ZMSK], 0)
    val rangeNear get() = (this.data[Op.MAXZ] and 0xFFFF).toDouble() / 65535.0
    val rangeFar get() = (this.data[Op.MINZ] and 0xFFFF).toDouble() / 65535.0
}

data class Color(var r: Float = 0f, var g: Float = 0f, var b: Float = 0f, var a: Float = 1f) {
    companion object {
        fun add(a: Color, b: Color, dest: Color = Color()): Color {
            dest.r = a.r + b.r
            dest.g = a.g + b.g
            dest.b = a.b + b.b
            dest.a = a.a * b.a
            return dest
        }
    }

    fun setRGB(rgb: Int): Color {
        this.r = rgb.extractScaledf01(0, 8).toFloat()
        this.g = rgb.extractScaledf01(8, 8).toFloat()
        this.b = rgb.extractScaledf01(16, 8).toFloat()
        this.a = 1f
        return this
    }

    fun setRGB_A(rgb: Int, a: Int) = this.apply {
        this.setRGB(rgb)
        this.a = rgb.extractScaledf01(0, 8).toFloat()
    }

    fun set(r: Float, g: Float, b: Float, a: Float = 1f) = this.apply {
        this.r = r
        this.g = g
        this.b = b
        this.a = a
    }

    fun equals(r: Float, g: Float, b: Float, a: Float): Boolean {
        return (this.r == r) && (this.g == g) && (this.b == b) && (this.a == a)
    }
}

class Blending(val data: IntArray) {
    private val fixColorSrc = Color()
    private val fixColorDst = Color()
    private val colorMaskColor = Color()

    val fixColorSource get() = fixColorSrc.setRGB(param24(this.data[Op.SFIX]))
    val fixColorDestination get() = fixColorDst.setRGB(param24(this.data[Op.DFIX]))
    val enabled get() = bool1(this.data[Op.ALPHABLENDENABLE])
    val functionSource get() = GuBlendingFactor(param4(this.data[Op.ALPHA], 0))
    val functionDestination get() = GuBlendingFactor(param4(this.data[Op.ALPHA], 4))
    val equation get() = GuBlendingEquation(param4(this.data[Op.ALPHA], 8))
    val colorMask get() = run { colorMaskColor.setRGB_A(param24(this.data[Op.PMSKC]), param8(this.data[Op.PMSKA], 0)) }
}

class AlphaTest(val data: IntArray) {
    val hash: Int get() = (this.data[Op.ALPHATESTENABLE] shl 24) or (this.data[Op.ATST])

    val enabled get() = bool1(this.data[Op.ALPHATESTENABLE])
    val func get() = TestFunctionEnum(param8(this.data[Op.ATST], 0))
    val value get() = param8(this.data[Op.ATST], 8)
    val mask get() = param8(this.data[Op.ATST], 16)
}

class Rectangle(val left: Int, val top: Int, val right: Int, val bottom: Int) {
    val width get() = this.right - this.left
    val height get() = this.bottom - this.top
}

class ClipPlane(val data: IntArray) {
    val enabled get() = bool1(this.data[Op.CLIPENABLE])
    val scissor get() = Rectangle(this.left, this.top, this.right, this.bottom)
    val left get() = param10(this.data[Op.SCISSOR1], 0)
    val top get() = param10(this.data[Op.SCISSOR1], 10)
    val right get() = param10(this.data[Op.SCISSOR2], 0)
    val bottom get() = param10(this.data[Op.SCISSOR2], 10)
}


class StencilState(val data: IntArray) {
    val enabled get() = bool1(this.data[Op.STENCILTESTENABLE])
    val fail get() = StencilOperationEnum(param8(this.data[Op.SOP], 0))
    val zfail get() = StencilOperationEnum(param8(this.data[Op.SOP], 8))
    val zpass get() = StencilOperationEnum(param8(this.data[Op.SOP], 16))
    val func get() = TestFunctionEnum(param8(this.data[Op.STST], 0))
    val funcRef get() = param8(this.data[Op.STST], 8)
    val funcMask get() = param8(this.data[Op.STST], 16)
}

class PatchState(val data: IntArray) {
    val divs get() = param8(this.data[Op.PATCHDIVISION], 0)
    val divt get() = param8(this.data[Op.PATCHDIVISION], 8)
}

class Fog(val data: IntArray) {
    val color get() = Color().setRGB(this.data[Op.FCOL])
    val far get() = float1(this.data[Op.FFAR])
    val dist get() = float1(this.data[Op.FDIST])
    val enabled get() = bool1(this.data[Op.FOGENABLE])
}

class LogicOp(val data: IntArray) {
    val enabled get() = this.data[Op.LOGICOPENABLE]
}

class LineSmoothState(val data: IntArray) {
    val enabled get() = bool1(this.data[Op.ANTIALIASENABLE])
}

class PatchCullingState(val data: IntArray) {
    val enabled get() = bool1(this.data[Op.PATCHCULLENABLE])
    val faceFlag get() = bool1(this.data[Op.PATCHFACING])
}

class OffsetState(val data: IntArray) {
    val x get() = param4(this.data[Op.OFFSETX], 0)
    val y get() = param4(this.data[Op.OFFSETY], 0)
}


// struct PspGeContext { unsigned int context[512] }
class GeState {
    companion object {
        const val STATE_NWORDS = 512
    }

    val data: IntArray = IntArray(STATE_NWORDS)

    val frameBuffer = GpuFrameBufferState(this.data)
    val vertex = VertexState(this.data)
    val stencil = StencilState(this.data)
    val viewport = ViewPort(this.data)
    val region = Region(this.data)
    val offset = OffsetState(this.data)
    val fog = Fog(this.data)
    val clipPlane = ClipPlane(this.data)
    val logicOp = LogicOp(this.data)
    val lightning = Lightning(this.data)
    val alphaTest = AlphaTest(this.data)
    val blending = Blending(this.data)
    val patch = PatchState(this.data)
    val texture = TextureState(this)
    val lineSmoothState = LineSmoothState(this.data)
    val patchCullingState = PatchCullingState(this.data)
    val culling = CullingState(this.data)
    val dithering = DitheringState(this.data)
    val colorTest = ColorTestState(this.data)
    val depthTest = DepthTestState(this.data)
    val skinning = SkinningState(this.data)
    //class SkinningState(val data: IntArray) {
    //	val boneMatrices = listOf(
    //		createMatrix4x3(this.data, Op.MAT_BONES + 12 * 0),
    //		createMatrix4x3(this.data, Op.MAT_BONES + 12 * 1),
    //		createMatrix4x3(this.data, Op.MAT_BONES + 12 * 2),
    //		createMatrix4x3(this.data, Op.MAT_BONES + 12 * 3),
    //		createMatrix4x3(this.data, Op.MAT_BONES + 12 * 4),
    //		createMatrix4x3(this.data, Op.MAT_BONES + 12 * 5),
    //		createMatrix4x3(this.data, Op.MAT_BONES + 12 * 6),
    //		createMatrix4x3(this.data, Op.MAT_BONES + 12 * 7)
    //	)
    //}

    val baseAddress: Int get() = (data[Op.BASE] shl 8) and 0xFF000000.toInt()

    var baseOffset: Int
        set(value) = run {
            data[Op.OFFSETADDR] = (data[Op.OFFSETADDR] and 0xFF000000.toInt()) or ((value ushr 8) and 0x00FFFFFF)
        }
        get() = data[Op.OFFSETADDR] shl 8

    fun writeInt(key: Int, offset: Int, value: Int): Unit = run { data[offset + data[key]++] = value }

    fun setTo(other: IntArray) = run { arraycopy(other, 0, this.data, 0, STATE_NWORDS) }
    fun setTo(other: GeState) = setTo(other.data)

    fun clone() = GeState().apply { setTo(this@GeState) }

    // Vertex
    val vertexType: Int get() = data[Op.VERTEXTYPE]
    val vertexReverseNormal: Boolean get() = data[Op.REVERSENORMAL] != 0
    var vertexAddress: Int
        set(value) = run { data[Op.VADDR] = setAddressRelativeToBaseOffset(data[Op.VADDR]) }
        get() = getAddressRelativeToBaseOffset(data[Op.VADDR])
    val indexAddress: Int get() = getAddressRelativeToBaseOffset(data[Op.IADDR])

    fun getAddressRelativeToBaseOffset(address: Int) = (baseAddress or address) + baseOffset
    fun setAddressRelativeToBaseOffset(address: Int) = (address and 0x00FFFFFF) - baseOffset
    fun getTextureMatrix(out: Matrix4 = Matrix4()) = out.apply { getMatrix4x4(this@GeState.data, Op.MAT_TEXTURE, out) }
    fun getProjMatrix(out: Matrix4 = Matrix4()) = out.apply { getMatrix4x4(this@GeState.data, Op.MAT_PROJ, out) }
    fun getViewMatrix(out: Matrix4 = Matrix4()) = out.apply { getMatrix4x3(this@GeState.data, Op.MAT_VIEW, out) }
    fun getWorldMatrix(out: Matrix4 = Matrix4()) = out.apply { getMatrix4x3(this@GeState.data, Op.MAT_WORLD, out) }
    fun getBoneMatrix(index: Int, out: Matrix4 = Matrix4()) =
        out.apply { getMatrix4x3(this@GeState.data, Op.MAT_BONES + 12 * index, out) }

    ///////////////////////////////

    //var baseOffset
    //	set(value) = run { this.data[Op.OFFSETADDR] = (this.data[Op.OFFSETADDR] and 0x00FFFFFF.inv()) or ((value ushr 8) and 0x00FFFFFF); }
    //	get() = param24(this.data[Op.OFFSETADDR]) shl 8

    val clearing get() = param1(this.data[Op.CLEAR], 0) != 0
    val clearFlags get() = param8(this.data[Op.CLEAR], 8)
    //val baseAddress get() = ((param24(this.data[Op.BASE]) shl 8) and 0xff000000.toInt())
    //val indexAddress get() = param24(this.data[Op.IADDR])
    val shadeModel get() = ShadingModelEnum(param16(this.data[Op.SHADEMODE], 0))
    val ambientModelColor get() = Color().setRGB_A(this.data[Op.MATERIALAMBIENT], this.data[Op.MATERIALALPHA])
    val diffuseModelColor get() = Color().setRGB(this.data[Op.MATERIALDIFFUSE])
    val specularModelColor get() = Color().setRGB(this.data[Op.MATERIALSPECULAR])
    val drawPixelFormat get() = PixelFormat(param4(this.data[Op.PSM], 0))

    //fun writeFloat(index: Int, offset: Int, data: Float) = this.dataf[offset + this.data[index]++] = data;

    fun getMorphWeight(index: Int) = float1(this.data[Op.MORPHWEIGHT0 + index])
    fun getAddressRelativeToBase(relativeAddress: Int) = (this.baseAddress or relativeAddress)
    fun reset() {
        data.fill(0)
    }
    //fun getAddressRelativeToBaseOffset(relativeAddress: Int) = ((this.baseAddress or relativeAddress) + this.baseOffset)

    ///////////////////////
}

class SkinningState(val data: IntArray) {
    fun getBoneMatrix(index: Int, out: Matrix4 = Matrix4()) =
        out.apply { getMatrix4x3(this@SkinningState.data, Op.MAT_BONES + 12 * index, out) }
}

private fun getFloat(value: Int) = Float.fromBits(value shl 8)

private fun getMatrix4x4(data: IntArray, register: Int, out: Matrix4) {
    for (n in 0 until 16) {
        out.data[n] = getFloat(data[register + n])
    }
}

private fun getMatrix4x3(data: IntArray, register: Int, out: Matrix4) {
    var m = 0
    var n = 0
    for (y in 0 until 4) {
        for (x in 0 until 3) {
            out.data[n + x] = getFloat(data[register + m])
            m++
        }
        n += 4
    }
    out.data[3] = 0f
    out.data[7] = 0f
    out.data[11] = 0f
    out.data[15] = 1f
}

class ColorTestState(val data: IntArray) {
    val enabled get() = bool1(this.data[Op.COLORTESTENABLE])
}

class DitheringState(val data: IntArray) {
    val enabled get() = bool1(this.data[Op.DITHERENABLE])
}

class GpuFrameBufferState(val data: IntArray) {
    val width get() = param16(this.data[Op.FRAMEBUFWIDTH], 0)
    val highAddress get() = param8(this.data[Op.FRAMEBUFWIDTH], 16)
    val lowAddress get() = param24(this.data[Op.FRAMEBUFPTR])
}

/**
 * Psp Component Order:
 * - weights, texture, color, normal, position
 */
class VertexType(v: Int = 0) {
    @NativeThreadLocal
    companion object {
        val DUMMY = VertexType()
    }

    private var computed = false
    var v: Int = v
        set(value) {
            if (field != value) {
                computed = false
                field = value
            }
        }

    var tex: NumericEnum get() = NumericEnum(v.extract(0, 2)); set(vv) = run { this.v = this.v.insert(vv.id, 0, 2) }
    var col: ColorEnum get() = ColorEnum(v.extract(2, 3)); set(vv) = run { this.v = this.v.insert(vv.id, 2, 3) }
    var normal: NumericEnum get() = NumericEnum(v.extract(5, 2)); set(vv) = run { this.v = this.v.insert(vv.id, 5, 2) }
    var pos: NumericEnum get() = NumericEnum(v.extract(7, 2)); set(vv) = run { this.v = this.v.insert(vv.id, 7, 2) }
    var weight: NumericEnum get() = NumericEnum(v.extract(9, 2)); set(vv) = run { this.v = this.v.insert(vv.id, 9, 2) }
    var index: IndexEnum get() = IndexEnum(v.extract(11, 2)); set(vv) = run { this.v = this.v.insert(vv.id, 11, 2) }
    var weightComponents: Int get() = v.extract(14, 3); set(vv) = run { this.v = this.v.insert(vv, 13, 3) }
    var morphingVertexCount: Int get() = v.extract(18, 2); set(vv) = run { this.v = this.v.insert(vv, 18, 2) }
    var transform2D: Boolean get() = v.extract(23, 1) != 0; set(vv) = run { this.v = this.v.insert(vv, 23) }

    val hasIndices: Boolean get() = index != IndexEnum.VOID
    val hasTexture: Boolean get() = tex != NumericEnum.VOID
    val hasColor: Boolean get() = col != ColorEnum.VOID
    val hasNormal: Boolean get() = normal != NumericEnum.VOID
    val hasPosition: Boolean get() = pos != NumericEnum.VOID
    val hasWeight: Boolean get() = weight != NumericEnum.VOID

    //val components: Int get() = if (transform2D) 2 else 3
    val components: Int get() = 3

    val posComponents: Int get() = components // @TODO: Verify this
    val normalComponents: Int get() = components // @TODO: Verify this
    val texComponents: Int get() = 2 // @TODO: texture components must be 2 or 3

    val colSize: Int get() = col.nbytes
    val normalSize: Int get() = normal.nbytes * normalComponents
    val positionSize: Int get() = pos.nbytes * posComponents
    val textureSize: Int get() = tex.nbytes * texComponents
    val weightSize: Int get() = weight.nbytes * weightComponents

    private var _colOffset: Int = 0
    private var _normalOffset: Int = 0
    private var _posOffset: Int = 0
    private var _texOffset: Int = 0
    private var _weightOffset: Int = 0
    private var _size: Int = 0

    val colOffset: Int get() = ensureComputed()._colOffset
    val normalOffset: Int get() = ensureComputed()._normalOffset
    val posOffset: Int get() = ensureComputed()._posOffset
    val texOffset: Int get() = ensureComputed()._texOffset
    val weightOffset: Int get() = ensureComputed()._weightOffset
    val size: Int get() = ensureComputed()._size

    private fun ensureComputed() = this.apply {
        if (!computed) {
            computed = true
            var out = 0
            out = out.nextAlignedTo(weight.nbytes); this._weightOffset = out; out = weightSize
            out = out.nextAlignedTo(tex.nbytes); this._texOffset = out; out += textureSize
            out = out.nextAlignedTo(col.nbytes); this._colOffset = out; out += colSize
            out = out.nextAlignedTo(normal.nbytes); this._normalOffset = out; out += normalSize
            out = out.nextAlignedTo(pos.nbytes); this._posOffset = out; out += positionSize
            this._size = out.nextAlignedTo(max(weight.nbytes, tex.nbytes, col.nbytes, normal.nbytes, pos.nbytes))
        }
    }

    fun init(v: Int) = this.apply {
        this.v = v
    }

    override fun toString(): String {
        val parts = arrayListOf<String>()
        parts += "color=$col"
        parts += "normal=$normal"
        parts += "pos=$pos"
        parts += "tex=$tex"
        parts += "weight=$weight"
        parts += "size=$size"
        return "VertexType(${parts.joinToString(", ")})"
    }
}

fun VertexType.init(state: GeState) = init(state.vertexType)

@Suppress("ArrayInDataClass")
data class VertexRaw(
    var type: VertexType = VertexType.DUMMY,
    var color: Int = 0,
    val normal: FloatArray = FloatArray(3),
    val pos: FloatArray = FloatArray(3),
    val tex: FloatArray = FloatArray(3),
    val weights: FloatArray = FloatArray(8)
) {
    override fun toString(): String {
        val parts = arrayListOf<String>()
        if (type.hasColor) parts += "color=${color.hex}"
        if (type.hasNormal) parts += "normal=${normal.toList()}"
        if (type.hasPosition) parts += "pos=${pos.toList()}"
        if (type.hasTexture) parts += "tex=${tex.toList()}"
        if (type.hasWeight) parts += "weights=${weights.toList()}"
        //"VertexRaw(${color.hex}, normal=${normal.toList()}, pos=${pos.toList()}, tex=${tex.toList()}, weights=${weights.toList()})"
        return "VertexRaw(${parts.joinToString(", ")})"
    }

    fun clone() = VertexRaw(
        type = VertexType.DUMMY,
        color = color,
        normal = normal.copyOf(),
        pos = pos.copyOf(),
        tex = tex.copyOf(),
        weights = weights.copyOf()
    )
}

class VertexReader {
    private fun SyncStream.readBytes(count: Int, out: FloatArray = FloatArray(4), normalized: Boolean) = out.apply {
        skipToAlign(4)
        if (normalized) {
            for (n in 0 until count) out[n] = readS8().toFloat() / 0x7F
        } else {
            for (n in 0 until count) out[n] = readS8().toFloat()
        }
    }

    private fun SyncStream.readShorts(count: Int, out: FloatArray = FloatArray(4), normalized: Boolean) = out.apply {
        skipToAlign(2)
        if (normalized) {
            for (n in 0 until count) out[n] = readS16_le().toFloat() / 0x7FFF
        } else {
            for (n in 0 until count) out[n] = readS16_le().toFloat()
        }
    }

    private fun SyncStream.readFloats(count: Int, out: FloatArray = FloatArray(4), normalized: Boolean) = out.apply {
        skipToAlign(4)
        for (n in 0 until count) out[n] = readF32_le()
    }

    fun SyncStream.readColorType(type: ColorEnum): RGBA {
        return when (type) {
            ColorEnum.COLOR4444 -> RGBA_4444.unpackToRGBA(readU16_le())
            ColorEnum.COLOR5551 -> RGBA_5551.unpackToRGBA(readU16_le())
            ColorEnum.COLOR5650 -> RGB_565.unpackToRGBA(readU16_le())
            ColorEnum.COLOR8888 -> RGBA(readS32_le())
            ColorEnum.VOID -> Colors.TRANSPARENT_BLACK
            else -> TODO()
        }
    }

    fun SyncStream.readNumericType(
        count: Int,
        type: NumericEnum,
        out: FloatArray = FloatArray(4),
        normalized: Boolean
    ): FloatArray = out.apply {
        when (type) {
            NumericEnum.VOID -> Unit
            NumericEnum.BYTE -> readBytes(count, out, normalized)
            NumericEnum.SHORT -> readShorts(count, out, normalized)
            NumericEnum.FLOAT -> readFloats(count, out, normalized)
        }
    }

    fun readOne(s: SyncStream, type: VertexType, out: VertexRaw = VertexRaw(type)): VertexRaw {
        out.type = type

        s.safeSkipToAlign(type.weight.nbytes)
        //println("Weight[0]: ${s.position} : align(${type.weight.nbytes})")
        s.readNumericType(type.weightComponents, type.weight, out.weights, normalized = true)
        //println("  Weight[1]: ${s.position}")

        s.safeSkipToAlign(type.tex.nbytes)
        //println("Tex[0]: ${s.position} : ${type.texComponents} : align(${type.tex.nbytes})")
        s.readNumericType(type.texComponents, type.tex, out.tex, normalized = true)
        //println("  Tex[1]: ${s.position}")

        s.safeSkipToAlign(type.col.nbytes)
        //println("Col[0]: ${s.position} : align(${type.col.nbytes})")
        out.color = s.readColorType(type.col).rgba
        //println("  Col[1]: ${s.position}")

        s.safeSkipToAlign(type.normal.nbytes)
        //println("Normal[0]: ${s.position} : ${type.normalComponents} : align(${type.normal.nbytes})")
        s.readNumericType(type.normalComponents, type.normal, out.normal, normalized = false)
        //println("  Normal[1]: ${s.position}")

        s.safeSkipToAlign(type.pos.nbytes)
        //println("Pos[0]: ${s.position} : ${type.posComponents} : align(${type.pos.nbytes})")
        s.readNumericType(type.posComponents, type.pos, out.pos, normalized = false)
        //println("  Pos[1]: ${s.position}")

        s.safeSkipToAlign(
            max(
                type.weight.nbytes,
                type.tex.nbytes,
                type.col.nbytes,
                type.normal.nbytes,
                type.pos.nbytes
            )
        )
        //println("Align: ${s.position}")

        return out
    }

    fun read(
        type: VertexType,
        count: Int,
        s: SyncStream,
        out: Array<VertexRaw> = Array(count) { VertexRaw() }
    ): Array<VertexRaw> {
        for (n in 0 until count) readOne(s, type, out[n])
        return out
    }

    fun readList(type: VertexType, count: Int, s: SyncStream) = read(type, count, s).toList()
}

private fun SyncStream.safeSkipToAlign(alignment: Int) = when {
    alignment == 0 -> Unit
    else -> this.skipToAlign(alignment)
}


