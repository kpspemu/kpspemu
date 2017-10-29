# kpspemu
PSP Emulator done in Kotlin Common targeting JVM and JS for now

### Previous works:
* https://github.com/soywiz/pspemu (PSP Emulator done in D programming language. Interpreted.)
* https://github.com/cspspemu/cspspemu (PSP Emulator done in C# programming language. Dynarec.)
* https://github.com/jspspemu/jspspemu (PSP Emulator done in typescript programming language. Dynarec.)

### Very basic online demo using Kotlin.JS:
* https://soywiz.github.io/kpspemu-demo (interpreted)
* http://jspspemu.com/#samples/minifire.elf (dynarec) <-- jspepmu

### Current state:

Right now, this is just a proof of concept. It just runs a very small demo in interpreted mode.

The aim is to create a portable emulator that can run fast in JVM, JS, Android, C++ targets (using libjit).

To achieve this, I have created an embedded module called `dynarek` that will provide an IR that
will generate JS code, JVM bytecode and relevant native code for each supported platform.

The rest of the code is kotlin common and uses [korge](https://github.com/korlibs/korge) and all
the [korlibs](https://github.com/korlibs/korlibs) libraries to do accelerated portable rendering.  