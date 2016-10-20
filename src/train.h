#ifndef __WORD2VEC_TRAIN_H__
# define __WORD2VEC_TRAIN_H__

//////////////
// Includes //
//////////////
#include "common.h"

////////////
// Macros //
////////////
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6

/////////////
// Structs //
/////////////


/////////////
// Methods //
/////////////

/**
 * Launch threads to train neural word embeddings on the specified file.
 *
 * @param a_opts - command line options defining training behavior
 * @type opt_t *
 *
 * @return \c void
 */
void train_model(opt_t *a_opts);
#endif  /* ifndef __WORD2VEC_TRAIN_H__ */
