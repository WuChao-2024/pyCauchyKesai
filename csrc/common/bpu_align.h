#ifndef BPU_ALIGN_H_
#define BPU_ALIGN_H_

// ALIGN_BPU: 根据编译时宏 HB_BPU_ALIGN 选择对齐值
// S100: 32, S600: 64, X5: 32
#ifndef HB_BPU_ALIGN
#error "HB_BPU_ALIGN must be defined by CMake (e.g. -DHB_BPU_ALIGN=64)"
#endif

#define ALIGN(value, alignment) (((value) + ((alignment)-1)) & ~((alignment)-1))
#define ALIGN_BPU(value) ALIGN(value, HB_BPU_ALIGN)

#endif  // BPU_ALIGN_H_
