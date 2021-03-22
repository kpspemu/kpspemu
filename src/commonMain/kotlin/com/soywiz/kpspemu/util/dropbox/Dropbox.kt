package com.soywiz.kpspemu.util.dropbox

import com.soywiz.klogger.*
import com.soywiz.kmem.*
import com.soywiz.korio.*
import com.soywiz.korio.async.*
import com.soywiz.korio.error.*
import com.soywiz.korio.file.*
import com.soywiz.korio.file.std.*
import com.soywiz.korio.lang.*
import com.soywiz.korio.net.http.*
import com.soywiz.korio.serialization.json.*
import com.soywiz.korio.stream.*
import com.soywiz.kpspemu.util.*
import kotlinx.coroutines.*
import kotlin.coroutines.*
import kotlin.math.*
import com.soywiz.korio.error.invalidOp
import kotlinx.coroutines.flow.*

// @TODO: Move to korio-ext-dropbox
class Dropbox(val bearer: String, val http: HttpClient = createHttpClient()) {
    val BASE_CONTENT = "https://content.dropboxapi.com"
    val BASE = "https://api.dropboxapi.com"


    companion object {
        val logger = Logger("Dropbox")

        // ...
        suspend fun login() {
        }
    }

    suspend fun rawRequest(
        path: String,
        content: AsyncStream? = null,
        headers: Http.Headers = Http.Headers(),
        method: Http.Method = Http.Method.POST,
        base: String = BASE
    ): HttpClient.Response {
        val fullUrl = "$base/$path"
        val allHeaders = Http.Headers(
            "Authorization" to "Bearer $bearer"
        ).withReplaceHeaders(headers.items)
        //println("### $fullUrl")
        //println(allHeaders)
        return http.request(method, fullUrl, allHeaders, content = content)
    }

    suspend fun jsonRawRequest(
        path: String,
        data: Map<String, Any>,
        headers: Http.Headers = Http.Headers(),
        method: Http.Method = Http.Method.POST,
        base: String = BASE
    ): DyAccess {
        val req = rawRequest(
            path = path,
            content = data.toJsonUntyped().openAsync(),
            headers = Http.Headers(
                "Content-Type" to "application/json"
            ).withReplaceHeaders(headers.items),
            method = method,
            base = base
        )
        if (!req.success) throw Http.HttpException(req.status, req.statusText, req.readAllString(), req.headers)
        //println(req.status)
        val res = req.readAllString()
        logger.trace { "jsonRawRequest.res:$res" }
        return Json.decode(res).dy
    }

    data class Entry(
        val name: String,
        val id: String,
        val tag: String,
        val size: Long,
        val contentHash: String
    ) {
        val isDirectory get() = tag == "folder"
        val isFile get() = !isDirectory
    }

    fun DyAccess.asMetadata() = Entry(
        name = this["name"].toString(),
        id = this["id"].toString(),
        tag = this[".tag"].toString(),
        size = this["size"].toLong(),
        contentHash = this["content_hash"].toString()
    )

    suspend fun listFolder(path: String): List<Entry> {
        val rpath = if (path == "/") "" else path
        val result = jsonRawRequest(
            "2/files/list_folder", hashMapOf<String, Any>(
                "path" to rpath,
                "recursive" to false
            )
        )
        logger.trace { "listFolder: $result" }
        return result["entries"].list().map { it.asMetadata() }
    }

    suspend fun createFolder(path: String): Unit {
        val rpath = if (path == "/") "" else path
        val result = jsonRawRequest(
            "2/files/create_folder", hashMapOf<String, Any>(
                "path" to rpath,
                "autorename" to false
            )
        )
        logger.trace { "create_folder: $result" }
    }

    suspend fun getMetadata(path: String): Entry {
        val rpath = if (path == "/") "" else path
        val result = jsonRawRequest(
            "2/files/get_metadata", hashMapOf<String, Any>(
                "path" to rpath
            )
        )
        logger.trace { "getMetadata: $result" }
        return result.asMetadata()
    }

    suspend fun download(path: String): AsyncInputStream {
        return rawRequest(
            "2/files/download", headers = Http.Headers(
                "Dropbox-API-Arg" to mapOf("path" to path).toJsonUntyped()
            ), base = BASE_CONTENT
        ).content
    }

    suspend fun downloadChunk(path: String, start: Long, size: Int): AsyncInputStream {
        return rawRequest(
            "2/files/download", headers = Http.Headers(
                "Dropbox-API-Arg" to mapOf("path" to path).toJsonUntyped(),
                "Range" to "bytes=$start-${start + size - 1}"
            ), base = BASE_CONTENT
        ).content
    }

    suspend fun upload(path: String, content: AsyncStream): Entry {
        val req = rawRequest(
            "2/files/upload", headers = Http.Headers(
                "Dropbox-API-Arg" to mapOf<String, Any>(
                    "path" to path,
                    "mode" to "overwrite",
                    "autorename" to false,
                    "mute" to false
                ).toJsonUntyped(),
                "Content-Type" to "application/octet-stream"
            ), content = content, base = BASE_CONTENT
        )

        return Json.parse(req.readAllString()).dy.asMetadata()
    }
}

fun <T : Comparable<T>> ClosedRange<T>.overlapsWith(that: ClosedRange<T>): Boolean =
    this.start <= that.endInclusive && that.start <= this.endInclusive

data class Option<out T>(val item: T?)

@Suppress("SortModifiers")
class DropboxVfs(val dropbox: Dropbox) : Vfs() {
    companion object {
        val logger = Logger("DropboxVfs")
    }

    val statCache = hashMapOf<String, Option<Dropbox.Entry>>()
    val pathCache = hashMapOf<String, List<String>>()

    fun String.normalizePath() = "/" + this.trim('/').replace("\\", "/")

    fun touched(path: String, created: Boolean) {
        statCache.remove(path.normalizePath())
        if (created) {
            pathCache.remove(PathInfo(path).folder.normalizePath())
        }
    }

    suspend override fun stat(path: String): VfsStat {
        val npath = path.normalizePath()
        val info = statCache.getOrPut(npath) { ignoreErrors { Option(dropbox.getMetadata(path)) } ?: Option(null) }
        val i = info.item

        return if (i != null) {
            createExistsStat(path, i.isDirectory, i.size)
        } else {
            createNonExistsStat(path)
        }


    }

    override suspend fun listSimple(path: String): List<VfsFile> {
        val npath = path.normalizePath()
        val names = pathCache.getOrPut(npath) { dropbox.listFolder(path).map { "$path/${it.name}" } }
        return names.map { file(it) }
    }

    suspend override fun mkdir(path: String, attributes: List<Attribute>): Boolean {
        println("Creating $path...")
        dropbox.createFolder(path)
        touched(path, created = true)
        println("Done Creating $path...")
        return true
    }

    suspend override fun open(path: String, mode: VfsOpenMode): AsyncStream {
        val info = ignoreErrors { dropbox.getMetadata(path) }
        if (info == null && !mode.createIfNotExists) invalidOp("File '$path' doesn't exists")
        var size = info?.size ?: 0L
        var fullContent: ByteArray? = null
        val patches = arrayListOf<Pair<Long, ByteArray>>()
        var setLengthPatch: Long? = null

        if (info == null) {
            // Upload empty file
            dropbox.upload(path, byteArrayOf().openAsync())
            touched(path, created = true)
        }

        // If the file is small <= 1MB. Read it directly!
        if (size <= 1L * 1024 * 1024) {
            fullContent = dropbox.download(path).readAll()
        }

        return object : AsyncStreamBase() {
            var consolidateTimer: Closeable? = null

            val consolidateQueue = AsyncQueue()

            fun mustConsolidate() = fullContent != null || patches.isNotEmpty() || setLengthPatch != null

            private suspend fun consolidateContent() = consolidateQueue {
                var changed = false

                if (fullContent == null) {
                    fullContent = dropbox.download(path).readAll()
                }

                if (patches.isNotEmpty()) {
                    fullContent = MemorySyncStreamToByteArray {
                        position = 0L
                        writeBytes(fullContent!!)
                        val apatches = patches.toList()
                        patches.clear()
                        for (patch in apatches) {
                            position = patch.first
                            writeBytes(patch.second)
                        }
                    }
                    changed = true
                }

                if (setLengthPatch != null) {
                    fullContent = fullContent!!.copyOf(setLengthPatch!!.toInt())
                    setLengthPatch = null
                    changed = true
                }

                size = fullContent!!.size.toLong()

                // Save to dropbox!
                logger.warn { "Must save to dropbox?" }
                if (changed) {
                    logger.warn { "Yes! It Changed" }
                    dropbox.upload(path, fullContent!!.openAsync())
                    touched(path, created = false)
                } else {
                    logger.warn { "No it didn't change!" }
                }
            }

            private suspend inline fun <T> addConsolidateTimer(callback: () -> T): T {
                val coroutineContext = coroutineContext
                consolidateTimer?.close()
                try {
                    return callback()
                } finally {

                    val task = asyncImmediately(coroutineContext) {
                        delay(200)
                        consolidateContent()
                    }

                    consolidateTimer = Closeable { task.cancel() }
                }
            }

            suspend override fun close() {
                consolidateContent()
            }

            suspend override fun getLength(): Long = size

            suspend override fun read(position: Long, buffer: ByteArray, offset: Int, len: Int): Int {
                if (mustConsolidate()) {
                    consolidateContent()
                    val content = fullContent!!
                    val available = content.size - position.toInt()
                    val read = min(available, len)
                    arraycopy(content, position.toInt(), buffer, offset, read)
                    return read
                } else {
                    val chunk = dropbox.downloadChunk(path, position, len).readAll()
                    arraycopy(chunk, 0, buffer, offset, chunk.size)
                    return chunk.size
                }
            }

            suspend override fun setLength(value: Long) {
                addConsolidateTimer {
                    setLengthPatch = value
                }
            }

            suspend override fun write(position: Long, buffer: ByteArray, offset: Int, len: Int) {
                addConsolidateTimer {
                    patches += position to buffer.copyOfRange(offset, offset + len)
                }
            }
        }.toAsyncStream()
    }
}

fun main(args: Array<String>) = Korio {
    val bearer = ApplicationDataVfs["config/dropbox.bearer"].readString().trim()
    val db = Dropbox(bearer)
    //for (entry in db.listFolder("/")) println(entry)
    //val bytes = db.download("/demo.png").readAll()
    //println(bytes.size)
    //val chunk = db.downloadChunk("/demo.png", 0L, 100).readAll()
    //println(chunk.size)
    println(db.getMetadata("/demo.png"))
    //println(bytes.toString(UTF8))
    //println(bearer)
}