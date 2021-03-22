package com.soywiz.kpspemu.util

import com.soywiz.korio.stream.*

fun SyncStream.sliceHere(): SyncStream = SyncStream(SliceSyncStreamBase(this.base, position, length))
