package com.soywiz.kpspemu.util

inline class DyAccess(val obj: Any?) : Iterable<DyAccess> {
    operator fun get(key: Int): DyAccess {
        return when (obj) {
            is List<*> -> obj[key]
            else -> null
        }.dy
    }

    operator fun get(key: String): DyAccess {
        return when (obj) {
            is Map<*, *> -> obj[key]
            else -> null
        }.dy
    }

    fun list(): List<DyAccess> {
        return when (obj) {
            is List<*> -> obj.map { it.dy }
        //is Map<*, *> -> obj[key]
            else -> listOf()
        }
    }

    val keys: List<DyAccess>
        get() = when (obj) {
            is Map<*, *> -> obj.keys.map { it.dy }
            else -> listOf()
        }

    fun toIntOrNull(): Int? = when (obj) {
        is Number -> obj.toInt()
        else -> obj.toString().toIntOrNull()
    }

    fun toLongOrNull(): Long? = when (obj) {
        is Number -> obj.toLong()
        else -> obj.toString().toLongOrNull()
    }

    fun toDoubleOrNull(): Double? = when (obj) {
        is Number -> obj.toDouble()
        else -> obj.toString().toDoubleOrNull()
    }

    override fun iterator(): Iterator<DyAccess> = list().iterator()

    fun toLong(default: Long = 0L): Long = toLongOrNull() ?: default
    fun toInt(default: Int = 0): Int = toIntOrNull() ?: default
    fun toDouble(default: Double = 0.0): Double = toDoubleOrNull() ?: default
    override fun toString(): String = obj.toString()
}

val Any?.dy get() = toDyAccess()
fun Any?.toDyAccess() = DyAccess(this)