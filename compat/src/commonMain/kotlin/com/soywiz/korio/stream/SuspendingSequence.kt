package com.soywiz.korio.stream

import com.soywiz.korio.async.*
import kotlinx.coroutines.channels.*
import kotlinx.coroutines.channels.toChannel

typealias SuspendingSequence<T> = ReceiveChannel<T>

suspend fun <T> Iterable<T>.toAsync() = this.toChannel()
