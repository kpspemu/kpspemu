package com.soywiz.kpspemu.util.dropbox

import com.soywiz.klogger.Logger
import com.soywiz.korio.Korio
import com.soywiz.korio.net.http.Http
import com.soywiz.korio.net.http.HttpClient
import com.soywiz.korio.net.http.createHttpClient
import com.soywiz.korio.serialization.json.Json
import com.soywiz.korio.serialization.json.toJsonUntyped
import com.soywiz.korio.stream.AsyncInputStream
import com.soywiz.korio.stream.AsyncStream
import com.soywiz.korio.stream.openAsync
import com.soywiz.korio.vfs.ApplicationDataVfs
import com.soywiz.kpspemu.util.DyAccess
import com.soywiz.kpspemu.util.dy

// @TODO: Move to korio-ext-dropbox
class Dropbox(val bearer: String) {
	val BASE_CONTENT = "https://content.dropboxapi.com"
	val BASE = "https://api.dropboxapi.com"
	val http = createHttpClient()

	companion object {
		val logger = Logger("Dropbox")

		// ...
		suspend fun login() {
		}
	}

	suspend fun rawRequest(path: String, content: AsyncStream? = null, headers: Http.Headers = Http.Headers(), method: Http.Method = Http.Method.POST, base: String = BASE): HttpClient.Response {
		val fullUrl = "$base/$path"
		val allHeaders = Http.Headers(
			"Authorization" to "Bearer $bearer"
		).withReplaceHeaders(headers.items)
		//println("### $fullUrl")
		//println(allHeaders)
		return http.request(method, fullUrl, allHeaders, content = content)
	}

	suspend fun jsonRawRequest(path: String, data: Map<String, Any>, headers: Http.Headers = Http.Headers(), method: Http.Method = Http.Method.POST, base: String = BASE): DyAccess {
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
		val folder: Boolean,
		val size: Long,
		val contentHash: String
	)

	fun DyAccess.asMetadata() = Entry(
		name = this["name"].toString(),
		id = this["id"].toString(),
		folder = this[".tag"].toString() == "folder",
		size = this["size"].toLong(),
		contentHash = this["content_hash"].toString()
	)

	suspend fun listFolder(path: String): List<Entry> {
		val rpath = if (path == "/") "" else path
		val result = jsonRawRequest("2/files/list_folder", hashMapOf<String, Any>(
			"path" to rpath,
			"recursive" to false
		))
		logger.trace { "listFolder: $result" }
		return result["entries"].list().map { it.asMetadata() }
	}

	suspend fun getMetadata(path: String): Entry {
		val rpath = if (path == "/") "" else path
		val result = jsonRawRequest("2/files/get_metadata", hashMapOf<String, Any>(
			"path" to rpath
		))
		logger.trace { "getMetadata: $result" }
		return result.asMetadata()
	}

	suspend fun download(path: String): AsyncInputStream {
		return rawRequest("2/files/download", headers = Http.Headers(
			"Dropbox-API-Arg" to mapOf("path" to path).toJsonUntyped()
		), base = BASE_CONTENT).content
	}

	suspend fun downloadChunk(path: String, start: Long, size: Int): AsyncInputStream {
		return rawRequest("2/files/download", headers = Http.Headers(
			"Dropbox-API-Arg" to mapOf("path" to path).toJsonUntyped(),
			"Range" to "bytes=$start-${start + size - 1}"
		), base = BASE_CONTENT).content
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