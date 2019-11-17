package com.soywiz.korma

import com.soywiz.korma.geom.*

typealias Matrix4 = Matrix3D
typealias Matrix2d = Matrix

fun Matrix4.setToIdentity() = identity()
fun Matrix4.setToMultiply(l: Matrix3D, r: Matrix3D) = multiply(l, r)

fun Matrix3D.setTo(
    a00: Float, a10: Float, a20: Float, a30: Float,
    a01: Float, a11: Float, a21: Float, a31: Float,
    a02: Float, a12: Float, a22: Float, a32: Float,
    a03: Float, a13: Float, a23: Float, a33: Float
) = this.setColumns(
    a00, a10, a20, a30,
    a01, a11, a21, a31,
    a02, a12, a22, a32,
    a03, a13, a23, a33
)
