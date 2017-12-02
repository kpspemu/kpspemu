package com.soywiz.kpspemu.ge

import com.soywiz.korma.Matrix4
import com.soywiz.korim.bitmap.Bitmap32
import com.soywiz.korma.Matrix2d
import com.soywiz.kpspemu.mem.Memory

data class GeBatchData(
	val state: IntArray,
	val primType: PrimitiveType,
	val vertexCount: Int,
	val vertices: ByteArray,
	val indices: ShortArray,
	val texVersion: Int = 0
)

class GeBatch {
	val vtype = VertexType()
	val state: GeState = GeState()
	lateinit var data: GeBatchData
	private val tempMatrix = Matrix4()
	val modelViewProjMatrix = Matrix4()
	val primType: PrimitiveType get() = data.primType
	val vertexCount: Int get() = data.vertexCount
	val vertices: ByteArray get() = data.vertices
	val indices: ShortArray get() = data.indices

	fun initData(data: GeBatchData) {
		this.data = data
		state.setTo(data.state)
		vtype.init(state)
		if (vtype.transform2D) {
			modelViewProjMatrix.setToOrtho(0f, 0f, 512f, 272f, 0f, (-0xFFFF).toFloat())
		} else {
			modelViewProjMatrix.setToIdentity()
			modelViewProjMatrix.setToMultiply(modelViewProjMatrix, state.getProjMatrix(tempMatrix))
			modelViewProjMatrix.setToMultiply(modelViewProjMatrix, state.getViewMatrix(tempMatrix))
			modelViewProjMatrix.setToMultiply(modelViewProjMatrix, state.getWorldMatrix(tempMatrix))
		}
	}

	fun hasTexture(): Boolean = state.texture.hasTexture

	fun getTextureId(): Int {
		val texture = state.texture
		if (texture.hasTexture) {
			return texture.mipmap.address
		}
		return 0
	}

	fun getTextureHash(mem: Memory): Int {
		var hash = 0
		val texture = state.texture
		if (texture.hasTexture) {
			val mipmap = texture.mipmaps[0]
			hash += mem.hash(mipmap.address, mipmap.sizeInBytes / 4)
			if (texture.hasClut) {
				val clut = texture.clut
				hash += mem.hash(clut.address, clut.sizeInBytes / 4)
			}
		}
		return hash
	}

	fun getTextureBitmap(mem: Memory): Bitmap32? {
		val texture = state.texture
		if (texture.hasTexture) {
			val mipmap = texture.mipmaps[0]
			val colorData = mem.readBytes(mipmap.address, mipmap.sizeInBytes)
			var clutData: ByteArray? = null
			var clutFormat: PixelFormat? = null
			var clutColors: Int = 0
			if (texture.hasClut) {
				val clut = texture.clut
				clutFormat = texture.clut.pixelFormat
				clutColors = texture.clut.numberOfColors
				clutData = mem.readBytes(clut.address, clut.sizeInBytes)
			}
			val texWidth = mipmap.bufferWidth
			val texHeight = mipmap.textureHeight

			return Bitmap32(texWidth, texHeight).setTo(texture.pixelFormat, colorData, clutData, clutFormat, clutColors, swizzled = texture.swizzled, width = texWidth, height = texHeight)
		} else {
			return null
		}
	}

	fun getEffectiveTextureMatrix(out: Matrix4 = Matrix4()): Matrix4 {
		//println(state.texture.textureMapMode)
		//println(vtype.transform2D)
		val transform = Matrix2d()

		if (vtype.transform2D) {
			val mipmap = state.texture.mipmap

			transform.setTransform(
				0.0, 0.0,
				1.0 / mipmap.bufferWidth.toDouble(), 1.0 / mipmap.textureHeight.toDouble(),
				0.0, 0.0, 0.0
			)
		} else {
			transform.setTransform(
				state.texture.offsetU.toDouble(), state.texture.offsetV.toDouble(),
				state.texture.scaleU.toDouble(), state.texture.scaleV.toDouble(),
				0.0, 0.0, 0.0
			)
		}

		transform.toMatrix4(out)
		return out
	}
}

fun Matrix2d.toMatrix4(out: Matrix4 = Matrix4()): Matrix4 = out.setTo(
	a.toFloat(), b.toFloat(), tx.toFloat(), 0f,
	c.toFloat(), d.toFloat(), ty.toFloat(), 0f,
	0f, 0f, 1f, 0f,
	0f, 0f, 0f, 1f
)