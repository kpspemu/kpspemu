package com.soywiz.dynarek2

import com.soywiz.dynarek2.target.jvm.*

actual fun D2Func.generate(context: D2Context, name: String?, debug: Boolean): D2Result = JvmGenerator().generate(this, context, name, debug)

actual fun D2Context.registerDefaultFunctions() {
}
