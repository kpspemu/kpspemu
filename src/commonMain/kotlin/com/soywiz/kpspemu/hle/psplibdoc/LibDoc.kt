package com.soywiz.kpspemu.hle.psplibdoc

import com.soywiz.korio.lang.*
import com.soywiz.korio.serialization.xml.*

object LibDoc {
    //data class Libraries(val libraries: List<Library>)

    interface Entry {
        val nid: Int
        val name: String
    }

    data class Function(override val nid: Int, override val name: String) : Entry
    data class Variable(override val nid: Int, override val name: String) : Entry
    data class Library(
        val name: String,
        val flags: Int,
        val functions: ArrayList<Function> = arrayListOf(),
        val variables: ArrayList<Variable> = arrayListOf()
    )

    data class Prx(val fileName: String, val name: String, val libraries: ArrayList<Library> = arrayListOf())
    data class Doc(val prxs: ArrayList<Prx> = arrayListOf()) {
        val allLibraries get() = prxs.flatMap { it.libraries }
    }

    fun parse(PSPLIBDOC: Xml): Doc {
        val doc = Doc()
        for (PRXFILE in PSPLIBDOC["PRXFILES"]["PRXFILE"]) {
            val prxFile = PRXFILE["prx"].first().text
            val prxName = PRXFILE["prxname"].first().text
            val prx = Prx(prxFile, prxName).apply { doc.prxs += this }
            //println("- prx=$prxFile, prxName=$prxName")
            for (LIBRARY in PRXFILE["LIBRARIES"]["LIBRARY"]) {
                val libraryName = LIBRARY["name"].first().text
                val libraryFlags = LIBRARY["flags"].first().text.parseInt()
                val library = Library(libraryName, libraryFlags.toInt()).apply { prx.libraries += this }
                //println("  - name=$libraryName, flags=$libraryFlags")
                for (FUNCTION in LIBRARY["FUNCTIONS"]["FUNCTION"]) {
                    val functionNid = FUNCTION["nid"].first().text.parseInt()
                    val functionName = FUNCTION["name"].first().text
                    val function = Function(functionNid, functionName).apply { library.functions += this }
                    //println("    - $function")
                }
                for (VARIABLE in LIBRARY["VARIABLES"]["VARIABLE"]) {
                    val variableNid = VARIABLE["nid"].first().text.parseInt()
                    val variableName = VARIABLE["name"].first().text
                    val variable = Variable(variableNid, variableName).apply { library.variables += this }
                    //println("    - $variable")
                }
            }
        }
        return doc
    }
}

fun LibDoc.Entry.dump() {
    println("    - $this")
}

fun LibDoc.Library.dump() {
    println("  - name=$name, flags=$flags")
    for (function in functions) function.dump()
    for (variable in variables) variable.dump()
}

fun LibDoc.Prx.dump() {
    println("- fileName=$fileName, name=$name")
    for (library in libraries) library.dump()
}

fun LibDoc.Doc.dump() {
    for (prx in prxs) prx.dump()
}