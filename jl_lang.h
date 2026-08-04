#ifndef JL_LANG_H
#define JL_LANG_H

#include "jl_core.h"

// Token types for the JL programming language
typedef enum {
    TOK_EOF,
    TOK_MICROKERNEL,   // microkernel <id>
    TOK_DISPATCH,       // dispatch <kernel_id> -> <target_kernel_id>
    TOK_QUANT_MATRIX,   // quant_matrix <name>
    TOK_ATTENTION,      // attention_head <dim>
    TOK_RING_SEND,      // ring_send <payload>
    TOK_INFER_LOOP,     // infer_loop <steps>
    TOK_IDENTIFIER,
    TOK_NUMBER,
    TOK_ARROW,          // ->
    TOK_LBRACE,         // {
    TOK_RBRACE          // }
} TokenType;

typedef struct {
    TokenType type;
    char text[64];
    int value;
} Token;

typedef struct {
    const char* source;
    size_t cursor;
    Token current_tok;
} Lexer;

// AST Nodes for the language compiler/interpreter
typedef enum {
    NODE_KERNEL_DEF,
    NODE_DISPATCH_CMD,
    NODE_MATRIX_ALLOC,
    NODE_INFER_LOOP
} ASTNodeType;

typedef struct ASTNode {
    ASTNodeType type;
    char name[64];
    int param1;
    int param2;
    struct ASTNode* next;
} ASTNode;

void lexer_init(Lexer* lexer, const char* source);
Token lexer_next(Lexer* lexer);
ASTNode* parse_jl_code(const char* source);
void execute_jl_ast(ASTNode* root, JlOsState* state);

#endif // JL_LANG_H
