package com.soywiz.kpspemu.cpu

object InstructionOpcodeDecoder {
    private data class Result(var result: InstructionType = Instructions.add)

    private class ResultInstructionEvaluator : InstructionEvaluator<Result>() {
        override fun unimplemented(s: Result, i: InstructionType) {
            s.result = i
        }
    }

    private val evaluator = ResultInstructionEvaluator()

    private val dispatcher = InstructionDispatcher(evaluator)

    operator fun invoke(i: Int): InstructionType = Result().run { dispatcher.dispatch(this, 0, i); this.result }
}
