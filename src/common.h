#ifndef __WORD2VEC_COMMON_H__
# define __WORD2VEC_COMMON_H__

//////////////
// Includes //
//////////////
#include <stdlib.h>

////////////
// Macros //
////////////
# define MAX_STRING 100

//////////////
// typedefs //
//////////////
typedef struct opt opt_t;
typedef float real;                    // Precision of float numbers

/////////////
// Structs //
/////////////

struct opt {
  char m_train_file[MAX_STRING];
  char m_output_file[MAX_STRING];

  long long m_layer1_size;
  long long m_iter;

  real m_alpha;
  real m_sample;

  int m_binary;
  int m_cbow;
  int m_debug_mode;
  int m_hs;
  int m_least_sq;
  int m_min_count;
  int m_min_reduce;
  int m_multitask;
  int m_negative;
  int m_num_threads;
  int m_task_specific;
  int m_window;
};

/////////////
// Methods //
/////////////

void reset_opt(opt_t *opt);

opt_t *create_opt(void);

#endif  /* ifndef __WORD2VEC_COMMON_H__ */
