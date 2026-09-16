/* Minimal jl_core.h for standalone inference testing */
#ifndef JL_CORE_H
#define JL_CORE_H

#include <stdint.h>
#include <stddef.h>

/* We're using libc now, not kernel malloc */
#include <stdlib.h>
#define kmalloc malloc
#define kfree free

typedef uint8_t  u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
typedef int8_t   s8;
typedef int16_t  s16;
typedef int32_t  s32;
typedef int64_t  s64;

#endif
