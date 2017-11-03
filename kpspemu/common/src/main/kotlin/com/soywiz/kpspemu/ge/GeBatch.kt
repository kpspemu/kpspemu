package com.soywiz.kpspemu.ge

import com.soywiz.korag.geom.Matrix4
import com.soywiz.korim.bitmap.Bitmap32
import com.soywiz.korma.Matrix2d
import com.soywiz.kpspemu.mem.Memory

data class GeBatch(
	val state: GeState,
	val primType: PrimitiveType,
	val vertexCount: Int,
	val vertices: ByteArray,
	val indices: ShortArray
) {
	val vtype = VertexType().init(state)
	private val tempMatrix = Matrix4()
	val modelViewProjMatrix = Matrix4()

	init {
		if (vtype.transform2D) {
			//modelViewProjMatrix.setToOrtho(0f, 480f, 272f, 0f, 0f, (-0xFFFF).toFloat())
			modelViewProjMatrix.setToOrtho(0f, 272f, 480f, 0f, 0f, (-0xFFFF).toFloat())
			//modelViewProjMatrix.setToIdentity()
		} else {
			modelViewProjMatrix.setToIdentity()
			modelViewProjMatrix.setToMultiply(modelViewProjMatrix, state.getProjMatrix(tempMatrix))
			modelViewProjMatrix.setToMultiply(modelViewProjMatrix, state.getViewMatrix(tempMatrix))
			modelViewProjMatrix.setToMultiply(modelViewProjMatrix, state.getWorldMatrix(tempMatrix))
		}
	}

	fun getTextureBitmap(mem: Memory): Bitmap32? {
		val texture = state.texture
		if (texture.hasTexture) {
			val mipmap = texture.mipmaps[0]
			val colorData = mem.readBytes(mipmap.address, mipmap.sizeInBytes)
			if (texture.hasClut) {
				val clut = texture.clut
				val clutData = mem.readBytes(clut.address, clut.sizeInBytes)
			}
			val texWidth = mipmap.bufferWidth
			val texHeight = mipmap.textureHeight
			return texture.pixelFormat.decode(Bitmap32(texWidth, texHeight), colorData)
		} else {
			return null
		}
	}

	fun getEffectiveTextureMatrix(out: Matrix4 = Matrix4()): Matrix4 {
		val transform = Matrix2d()
		transform.setTransform(
			state.texture.offsetU.toDouble(), state.texture.offsetV.toDouble(),
			state.texture.scaleU.toDouble(), state.texture.scaleV.toDouble(),
			0.0, 0.0, 0.0
		)
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