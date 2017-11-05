import com.soywiz.korio.vfs.localCurrentDirVfs

object KpspTests {
	val root = localCurrentDirVfs["../.."].jail()
	val pspautotests = root["pspautotests"]
	val samples = root["samples"]
	val rootTestResources = root["kpspemu/common/src/test/resources"]
}
