/* ============================================================================
 * JL OS: CUSTOM LANGUAGE DRIVER & VM TEST RUNNER
 * ============================================================================ */

#include <stdio.h>
#include "jl_core.h"
#include "jl_lang.h"

// Sample code written directly in our new custom "JL" programming language!
const char* JL_SCRIPT = 
    "microkernel MK_TENSOR_CORE\n"
    "microkernel MK_ATTN_SCHED\n"
    "quant_matrix WEIGHT_LAYER_0 64\n"
    "dispatch MK_1 -> MK_2\n"
    "dispatch MK_2 -> MK_3\n"
    "infer_loop 8\n";

int main(void) {
    printf("================================================================================\n");
    printf("=== [JL OS OS] CUSTOM PROGRAMMING LANGUAGE INTERPRETER & COMPILER ======\n");
    printf("================================================================================\n");
    
    JlOsState state;
    jl_os_init(&state);

    printf("Parsing Custom Source Script written in 'JL-Lang':\n");
    printf("--------------------------------------------------------------------------------\n");
    printf("%s", JL_SCRIPT);
    printf("--------------------------------------------------------------------------------\n");

    ASTNode* ast = parse_jl_code(JL_SCRIPT);
    execute_jl_ast(ast, &state);

    return 0;
}
