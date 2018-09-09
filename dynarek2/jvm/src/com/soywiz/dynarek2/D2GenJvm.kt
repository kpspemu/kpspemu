package com.soywiz.dynarek2

import com.soywiz.dynarek2.target.jvm.*

actual fun D2Func.generate(): D2Result = JvmGenerator().generate(this)
