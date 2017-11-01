package com.soywiz.kpspemu.util

import com.soywiz.korio.lang.Console

object PspLoggerManager {
	val loggers = LinkedHashMap<String, PspLogger>()
	var defaultLevel: PspLogLevel? = null

	fun getLogger(name: String) = loggers.getOrPut(name) { PspLogger(name, true) }

	fun setLevel(name: String, level: PspLogLevel) = getLogger(name).apply { this.level = level }
}

enum class PspLogLevel(val index: Int) { NONE(0), ERROR(1), WARN(2), INFO(3), TRACE(4) }

class PspLogger internal constructor(val name: String, val dummy: Boolean) {
	companion object {
		operator fun invoke(name: String) = PspLoggerManager.getLogger(name)
	}

	init {
		PspLoggerManager.loggers[name] = this
	}

	var level: PspLogLevel? = null

	val processedLevel: PspLogLevel get() = level ?: PspLoggerManager.defaultLevel ?: PspLogLevel.WARN

	@PublishedApi
	internal fun actualLog(level: PspLogLevel, msg: String) {
		val line = "[$name]: $msg"
		when (level) {
			PspLogLevel.ERROR -> Console.error(line)
			else -> Console.log(line)
		}
	}

	inline fun log(level: PspLogLevel, msg: () -> String) {
		if (level.index <= processedLevel.index) {
			actualLog(level, msg())
		}
	}

	fun error(msg: String) = log(PspLogLevel.ERROR) { msg }
	fun warn(msg: String) = log(PspLogLevel.WARN) { msg }
	fun info(msg: String) = log(PspLogLevel.INFO) { msg }
	fun trace(msg: String) = log(PspLogLevel.TRACE) { msg }

	inline fun error(msg: () -> String) = log(PspLogLevel.ERROR, msg)
	inline fun warn(msg: () -> String) = log(PspLogLevel.WARN, msg)
	inline fun info(msg: () -> String) = log(PspLogLevel.INFO, msg)
	inline fun trace(msg: () -> String) = log(PspLogLevel.TRACE, msg)
}