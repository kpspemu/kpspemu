package com.soywiz.dynarek2

import com.soywiz.dynarek2.target.jvm.*

actual fun D2Func.generate(name: String?, debug: Boolean): D2Result = JvmGenerator().generate(this, name, debug)
