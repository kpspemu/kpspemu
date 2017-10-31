package com.soywiz.kpspemu.util

import com.soywiz.korio.lang.Console

object LoggerManager {
	val loggers = LinkedHashMap<String, Logger>()
	var allLevels: LogLevel? = null

	fun getLogger(name: String) = loggers.getOrPut(name) { Logger(name, true) }
}

enum class LogLevel(val index: Int) { NONE(0), ERROR(1), WARN(2), INFO(3), TRACE(4) }

class Logger internal constructor(val name: String, val dummy: Boolean) {
	companion object {
		operator fun invoke(name: String) = LoggerManager.getLogger(name)
	}

	init {
		LoggerManager.loggers[name] = this
	}

	var level = LogLevel.WARN

	val processedLevel: LogLevel get() = LoggerManager.allLevels ?: level

	fun log(level: LogLevel, msg: Any?) {
		if (level.index <= processedLevel.index) {
			if (level == LogLevel.ERROR) {
				Console.error(msg)
			} else {
				Console.log(msg)
			}
		}
	}

	fun error(msg: Any?) = log(LogLevel.ERROR, msg)
	fun warn(msg: Any?) = log(LogLevel.WARN, msg)
	fun info(msg: Any?) = log(LogLevel.INFO, msg)
	fun trace(msg: Any?) = log(LogLevel.TRACE, msg)
}