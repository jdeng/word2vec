#ifndef __WORD2VEC_IO_H__
# define __WORD2VEC_IO_H__

//////////////
// Includes //
//////////////
#include "common.h"
#include "vocab.h"

#include <stdio.h>   /* fopen, getline, ferror */

/////////////
// Methods //
/////////////

/**
 * Reads a single word from a file.
 *
 * @param word - target word to populate
 * @type char *
 * @param fin - input stream
 * @type FILE *
 *
 * @return \c void
 */
void ReadWord(char *word, FILE *fin);

/**
 * Read a word and return its index in the vocabulary.
 *
 * @param fin - input stream
 * @type FILE *
 *
 * @return \c void
 */
int ReadWordIndex(FILE *fin);

/**
 * Create vocabulary for words in the training file.
 *
 * @param a_vocab - vocabulary to populate
 * @type vocab_t *
 * @param a_opts - word to search for
 * @type const opt_t *
 *
 * @return \c void
 */
void learn_vocab_from_trainfile(vocab_t *a_vocab, const opt_t *a_opts);
#endif  /* ifndef __WORD2VEC_IO_H__ */
