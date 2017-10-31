package com.soywiz.kpspemu.ge

import com.soywiz.korag.geom.Matrix4

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
			modelViewProjMatrix.setToOrtho(0f, 480f, 272f, 0f, 0f, (-0xFFFF).toFloat())
		} else {
			modelViewProjMatrix.setToIdentity()
			modelViewProjMatrix.setToMultiply(modelViewProjMatrix, state.getProjMatrix(tempMatrix))
			modelViewProjMatrix.setToMultiply(modelViewProjMatrix, state.getViewMatrix(tempMatrix))
			modelViewProjMatrix.setToMultiply(modelViewProjMatrix, state.getWorldMatrix(tempMatrix))
		}
	}
}