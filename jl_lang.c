/* ============================================================================
 * JL OS: CUSTOM PROGRAMMING LANGUAGE COMPILER & VM RUNTIME
 * ============================================================================ */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "jl_lang.h"

static int is_alpha(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
}

static int is_digit(char c) {
    return c >= '0' && c <= '9';
}

void lexer_init(Lexer* lexer, const char* source) {
    lexer->source = source;
    lexer->cursor = 0;
}

Token lexer_next(Lexer* lexer) {
    Token tok;
    tok.type = TOK_EOF;
    tok.text[0] = '\0';
    tok.value = 0;

    // Skip whitespace
    while (lexer->source[lexer->cursor] == ' ' || 
           lexer->source[lexer->cursor] == '\t' || 
           lexer->source[lexer->cursor] == '\n' || 
           lexer->source[lexer->cursor] == '\r') {
        lexer->cursor++;
    }

    char c = lexer->source[lexer->cursor];
    if (c == '\0') return tok;

    if (c == '-' && lexer->source[lexer->cursor + 1] == '>') {
        tok.type = TOK_ARROW;
        strcpy(tok.text, "->");
        lexer->cursor += 2;
        return tok;
    }

    if (c == '{') {
        tok.type = TOK_LBRACE;
        tok.text[0] = '{'; tok.text[1] = '\0';
        lexer->cursor++;
        return tok;
    }

    if (c == '}') {
        tok.type = TOK_RBRACE;
        tok.text[0] = '}'; tok.text[1] = '\0';
        lexer->cursor++;
        return tok;
    }

    if (is_digit(c)) {
        size_t len = 0;
        while (is_digit(lexer->source[lexer->cursor])) {
            tok.text[len++] = lexer->source[lexer->cursor++];
        }
        tok.text[len] = '\0';
        tok.type = TOK_NUMBER;
        tok.value = atoi(tok.text);
        return tok;
    }

    if (is_alpha(c)) {
        size_t len = 0;
        while (is_alpha(lexer->source[lexer->cursor]) || is_digit(lexer->source[lexer->cursor])) {
            tok.text[len++] = lexer->source[lexer->cursor++];
        }
        tok.text[len] = '\0';

        if (strcmp(tok.text, "microkernel") == 0) tok.type = TOK_MICROKERNEL;
        else if (strcmp(tok.text, "dispatch") == 0) tok.type = TOK_DISPATCH;
        else if (strcmp(tok.text, "quant_matrix") == 0) tok.type = TOK_QUANT_MATRIX;
        else if (strcmp(tok.text, "ring_send") == 0) tok.type = TOK_RING_SEND;
        else if (strcmp(tok.text, "infer_loop") == 0) tok.type = TOK_INFER_LOOP;
        else tok.type = TOK_IDENTIFIER;

        return tok;
    }

    lexer->cursor++;
    return tok;
}

ASTNode* parse_jl_code(const char* source) {
    Lexer lexer;
    lexer_init(&lexer, source);

    ASTNode* head = NULL;
    ASTNode* tail = NULL;

    Token tok = lexer_next(&lexer);
    while (tok.type != TOK_EOF) {
        ASTNode* node = (ASTNode*)malloc(sizeof(ASTNode));
        memset(node, 0, sizeof(ASTNode));

        if (tok.type == TOK_MICROKERNEL) {
            node->type = NODE_KERNEL_DEF;
            Token id_tok = lexer_next(&lexer);
            strcpy(node->name, id_tok.text);
        } 
        else if (tok.type == TOK_DISPATCH) {
            node->type = NODE_DISPATCH_CMD;
            Token src_tok = lexer_next(&lexer);
            strcpy(node->name, src_tok.text);
            
            lexer_next(&lexer); // arrow ->
            Token dst_tok = lexer_next(&lexer);
            node->param1 = dst_tok.text[3] - '0'; // extract kernel ID number
        } 
        else if (tok.type == TOK_QUANT_MATRIX) {
            node->type = NODE_MATRIX_ALLOC;
            Token name_tok = lexer_next(&lexer);
            strcpy(node->name, name_tok.text);
            Token dim_tok = lexer_next(&lexer);
            node->param1 = dim_tok.value;
        }
        else if (tok.type == TOK_INFER_LOOP) {
            node->type = NODE_INFER_LOOP;
            Token count_tok = lexer_next(&lexer);
            node->param1 = count_tok.value;
        }

        if (!head) {
            head = node;
            tail = node;
        } else {
            tail->next = node;
            tail = node;
        }

        tok = lexer_next(&lexer);
    }

    return head;
}

void execute_jl_ast(ASTNode* root, JlOsState* state) {
    printf("\n================================================================================\n");
    printf("[JL OS COMPILER & VM RUNTIME] EXECUTING CUSTOM CODE SCRIPT\n");
    printf("================================================================================\n");

    ASTNode* curr = root;
    while (curr) {
        switch (curr->type) {
            case NODE_KERNEL_DEF:
                printf("[JL-LANG] Spawned Custom Microkernel Definition: '%s'\n", curr->name);
                break;
            case NODE_MATRIX_ALLOC:
                printf("[JL-LANG] Allocating Quantized Tensors '%s' (Dimension: %d x %d)\n", 
                       curr->name, curr->param1, curr->param1);
                break;
            case NODE_DISPATCH_CMD:
                printf("[JL-LANG] Low-Level Dispatch: %s -> Target Core #%d\n", curr->name, curr->param1);
                break;
            case NODE_INFER_LOOP:
                printf("[JL-LANG] Running Custom Inference Loop (%d steps):\n", curr->param1);
                char token[64];
                for (int i = 0; i < curr->param1; i++) {
                    jl_inference_step(state, token);
                    printf("  └─> [LANG TOKEN OUT]: %s\n", token);
                }
                break;
        }
        curr = curr->next;
    }
    printf("================================================================================\n");
}
