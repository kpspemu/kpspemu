package com.soywiz.korio.serialization.json

fun Json.decode(str: String): Any? = this.parse(str)

fun Map<*, *>.toJsonUntyped() = this.toJson()