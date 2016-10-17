//////////////
// Includes //
//////////////
#include "common.h"

/////////////
// Methods //
/////////////

void reset_opt(opt_t *opt) {
  opt->m_train_file[0] = 0;
  opt->m_output_file[0] = 0;

  opt->m_layer1_size = 100;
  opt->m_iter = 5;

  opt->m_alpha = 0.025;
  opt->m_sample = 1e-3;

  opt->m_binary = 0;
  opt->m_cbow = 1;
  opt->m_debug_mode = 2;
  opt->m_hs = 0;
  opt->m_least_sq = 0;
  opt->m_min_count = 5;
  opt->m_min_reduce = 1;
  opt->m_multitask = 0;
  opt->m_negative = 5;
  opt->m_num_threads = 12;
  opt->m_task_specific = 0;
  opt->m_window = 5;
}

opt_t *create_opt(void) {
  opt_t *opts = calloc(1, sizeof(opt_t));
  reset_opt(opts);
  return opts;
}
