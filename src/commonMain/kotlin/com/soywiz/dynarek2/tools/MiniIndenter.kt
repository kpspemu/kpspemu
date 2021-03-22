package com.soywiz.dynarek2.tools

class MiniIndenter {
    object INDENT
    object UNINDENT

    val commands = arrayListOf<Any>()

    fun line(content: MiniIndenter) {
        commands.addAll(content.commands)
    }

    fun line(content: String) {
        commands += content
    }

    inline fun line(content: String, callback: () -> Unit) {
        commands += "$content {"
        indent(callback)
        commands += "}"
    }

    fun indent() {
        commands += INDENT
    }

    fun unindent() {
        commands += UNINDENT
    }

    inline fun indent(callback: () -> Unit) {
        indent()
        try {
            callback()
        } finally {
            unindent()
        }
    }

    override fun toString(): String = buildString {
        var nindent = 0
        for (cmd in commands) {
            when (cmd) {
                INDENT -> nindent++
                UNINDENT -> nindent--
                else -> {
                    append(INDENTS[nindent])
                    append(cmd.toString())
                    append('\n')
                }
            }
        }
    }

    companion object {
        operator fun invoke(callback: MiniIndenter.() -> Unit): MiniIndenter = MiniIndenter().apply(callback)
    }

    object INDENTS {
        private val INDENTS: Array<String> = Array(1024) { "" }.apply {
            val indent = StringBuilder()
            for (n in 0 until this.size) {
                this[n] = indent.toString()
                indent.append("\t")
            }
        }

        operator fun get(index: Int): String {
            if (index >= INDENTS.size) TODO("Too much indentation $index")
            return if (index <= 0) "" else INDENTS[index]
        }
    }
}
