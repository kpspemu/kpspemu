package com.soywiz.kpspemu.util

import com.soywiz.korio.stream.SliceSyncStreamBase
import com.soywiz.korio.stream.SyncStream

fun SyncStream.sliceHere(): SyncStream = SyncStream(SliceSyncStreamBase(this.base, position, length))
