import com.soywiz.korio.file.std.*

object KpspTests {
    val root = localCurrentDirVfs["../.."].jail()
    val pspautotests = root["pspautotests"]
    val rootTestResources = root["kpspemu/common/src/test/resources"]
}
