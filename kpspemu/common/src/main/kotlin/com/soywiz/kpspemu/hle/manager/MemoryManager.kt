@file:Suppress("CanBeVal")

package com.soywiz.kpspemu.hle.manager

import com.soywiz.korio.lang.format
import com.soywiz.kpspemu.util.reduceAcumulate
import com.soywiz.kpspemu.util.splice

class MemoryManager {
	val memoryPartitionsUid: MutableMap<Int, MemoryPartition> = LinkedHashMap<Int, MemoryPartition>()

	init {
		this.memoryPartitionsUid[MemoryPartitions.Kernel0] = MemoryPartition("Kernel Partition 1", 0x88000000, 0x88300000, false)
		//this.memoryPartitionsUid[MemoryPartitions.User] = new MemoryPartition("User Partition", 0x08800000, 0x08800000 + 0x100000 * 32, false);
		//this.memoryPartitionsUid[MemoryPartitions.UserStacks] = new MemoryPartition("User Stacks Partition", 0x08800000, 0x08800000 + 0x100000 * 32, false);
		this.memoryPartitionsUid[MemoryPartitions.User] = MemoryPartition("User Partition", 0x08800000, 0x08800000 + 0x100000 * 24, false)
		this.memoryPartitionsUid[MemoryPartitions.UserStacks] = MemoryPartition("User Stacks Partition", 0x08800000, 0x08800000 + 0x100000 * 24, false)
		this.memoryPartitionsUid[MemoryPartitions.VolatilePartition] = MemoryPartition("Volatile Partition", 0x08400000, 0x08800000, false)
	}

	val kernelPartition: MemoryPartition get() = this.memoryPartitionsUid[MemoryPartitions.Kernel0]!!
	val userPartition: MemoryPartition get() = this.memoryPartitionsUid[MemoryPartitions.User]!!
	val stackPartition: MemoryPartition get() = this.memoryPartitionsUid[MemoryPartitions.UserStacks]!!
}

open class MemoryPartitions(val id: Int) {
	companion object {
		const val Kernel0 = 0
		const val User = 2
		const val VolatilePartition = 5
		const val UserStacks = 6
	}
}

enum class MemoryAnchor(val id: Int) {
	Low(0),
	High(1),
	Address(2),
	LowAligned(3),
	HighAligned(4),
}

class OutOfMemoryError(message: String) : Exception(message)

data class MemoryPartition(
	var name: String,
	val low: Double,
	val high: Double,
	var allocated: Boolean,
	val parent: MemoryPartition? = null
) {
	companion object {
		inline operator fun invoke(
			name: String,
			low: Number,
			high: Number,
			allocated: Boolean,
			parent: MemoryPartition? = null
		) = MemoryPartition(name, low.toDouble(), high.toDouble(), allocated, parent)
	}

	// Actual address
	var address: Double = low

	private val _childPartitions = arrayListOf<MemoryPartition>()

	val free: Boolean get() = !allocated

	val size: Double get() = this.high - this.low
	val root: MemoryPartition get() = this.parent?.root ?: this

	val childPartitions: ArrayList<MemoryPartition>
		get() {
			if (this._childPartitions.isEmpty()) {
				this._childPartitions.add(MemoryPartition("", this.low, this.high, false, this))
			}
			return this._childPartitions
		}

	fun contains(address: Double): Boolean = address >= this.low && address < this.high

	fun deallocate() {
		this.allocated = false
		this.parent?.cleanup()
	}

	fun allocate(size: Double, anchor: MemoryAnchor, address: Double = 0.0, name: String = ""): MemoryPartition {
		when (anchor) {
			MemoryAnchor.LowAligned, // @TODO: aligned!
			MemoryAnchor.Low -> return this.allocateLow(size, name)
			MemoryAnchor.High -> return this.allocateHigh(size, name)
			MemoryAnchor.Address -> return this.allocateSet(size, address, name)
			else -> throw Error("Not implemented anchor %d:%s".format(anchor, anchor))
		}
	}

	inline fun allocateSet(size: Number, addressLow: Number, name: String = ""): MemoryPartition = allocateSet(size.toDouble(), addressLow.toDouble(), name)
	inline fun allocateLow(size: Number, name: String = ""): MemoryPartition = this.allocateLow(size.toDouble(), name)
	inline fun allocateHigh(size: Number, name: String = "", alignment: Int = 1): MemoryPartition = this.allocateHigh(size.toDouble(), name)

	fun allocateSet(size: Double, addressLow: Double, name: String = ""): MemoryPartition {
		var childs = this.childPartitions
		var addressHigh = addressLow + size

		if (!this.contains(addressLow) || !this.contains(addressHigh)) {
			throw OutOfMemoryError("Can't allocate [%08X-%08X] in [%08X-%08X]".format(addressLow, addressHigh, this.low, this.high))
		}

		val index = childs.indexOfFirst { it.contains(addressLow) }
		if (index < 0) {
			println("address: %08X, size: %d".format(addressLow, size))
			println(this)
			throw Error("Can't find the segment")
		}

		var child = childs[index]
		if (child.allocated) throw Error("Memory already allocated")
		if (!child.contains(addressHigh - 1)) throw Error("Can't fit memory")

		var p1 = MemoryPartition("", child.low, addressLow, false, this)
		var p2 = MemoryPartition(name, addressLow, addressHigh, true, this)
		var p3 = MemoryPartition("", addressHigh, child.high, false, this)

		childs.splice(index, 1, p1, p2, p3)

		this.cleanup()
		return p2
	}

	fun allocateLow(size: Double, name: String = ""): MemoryPartition = this.allocateLowHigh(size, true, name)
	fun allocateHigh(size: Double, name: String = "", alignment: Int = 1): MemoryPartition = this.allocateLowHigh(size, false, name)

	private fun _validateChilds() {
		var childs = this.childPartitions

		if (childs[0].low != this.low) throw Error("First child low doesn't match container low")
		if (childs[childs.size - 1].high != this.high) throw Error("Last child high doesn't match container high")

		for (n in 0 until childs.size - 1) {
			if (childs[n + 0].high != childs[n + 1].low) throw Error("Children at $n are not contiguous")
		}
	}

	private fun allocateLowHigh(size: Double, low: Boolean, name: String = ""): MemoryPartition {
		var childs = this.childPartitions

		val index = childs.indexOfFirst { it.free && it.size >= size }
		if (index < 0) throw OutOfMemoryError("Can't find a partition with $size available")
		var child = childs[index]

		val unallocatedChild: MemoryPartition
		val allocatedChild: MemoryPartition

		if (low) {
			var p1 = child.low
			var p2 = child.low + size
			var p3 = child.high
			allocatedChild = MemoryPartition(name, p1, p2, true, this)
			unallocatedChild = MemoryPartition("", p2, p3, false, this)
			childs.splice(index, 1, allocatedChild, unallocatedChild)
		} else {
			var p1 = child.low
			var p2 = child.high - size
			var p3 = child.high
			unallocatedChild = MemoryPartition("", p1, p2, false, this)
			allocatedChild = MemoryPartition(name, p2, p3, true, this)
			childs.splice(index, 1, unallocatedChild, allocatedChild)
		}

		this.cleanup()
		return allocatedChild
	}

	fun unallocate() {
		this.name = ""
		this.allocated = false
		this.parent?.cleanup()
	}

	private fun cleanup() {
		var startTotalFreeMemory = this.getTotalFreeMemory()

		this._validateChilds()

		// join contiguous free memory
		var childs = this.childPartitions
		if (childs.size >= 2) {
			var n = 0
			while (n < childs.size - 1) {
				var l = childs[n + 0]
				var r = childs[n + 1]
				if (!l.allocated && !r.allocated) {
					val new = MemoryPartition("", l.low, r.high, false, this)
					//console.log('joining', child, c1, child.low, c1.high);
					childs.splice(n, 2, new)
					//println("l: $l")
					//println("r: $r")
					//println("new: $new")
				} else {
					n++
				}
			}
		}
		// remove empty segments
		run {
			var n = 0
			while (n < childs.size) {
				var child = childs[n]
				if (!child.allocated && child.size == 0.0) {
					childs.splice(n, 1)
				} else {
					n++
				}
			}
		}

		this._validateChilds()

		var endTotalFreeMemory = this.getTotalFreeMemory()

		if (endTotalFreeMemory != startTotalFreeMemory) {
			println("assertion failed [1]! : $startTotalFreeMemory,$endTotalFreeMemory")
		}
	}

	val nonAllocatedPartitions get() = this.childPartitions.filter { !it.allocated }
	fun getTotalFreeMemory(): Double = this.nonAllocatedPartitions.reduceAcumulate(0.0) { prev, item -> item.size + prev }
	fun getMaxContiguousFreeMemory(): Double = this.nonAllocatedPartitions.maxBy { it.size }?.size ?: 0.0

	fun getTotalFreeMemoryInt(): Int = this.getTotalFreeMemory().toInt()
	fun getMaxContiguousFreeMemoryInt(): Int = this.getMaxContiguousFreeMemory().toInt()

	private fun findFreeChildWithSize(size: Int) = Unit
}

