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
 * @param a_vocab - vocabulary to search in
 * @type vw_t *
 * @param a_vocab_hash - hash of word indices
 * @type int *
 *
 * @return \c void
 */
int ReadWordIndex(FILE *fin, const vw_t *a_vocab, const int *a_vocab_hash);

/**
 * Create vocabulary from words in the training file.
 *
 * @param a_vocab - vocabulary to populate
 * @type vocab_t *
 * @param a_opts - word to search for
 * @type opt_t *
 *
 * @return \c size_t - size of the input file
 */
size_t learn_vocab_from_trainfile(vocab_t *a_vocab, opt_t *a_opts);
#endif  /* ifndef __WORD2VEC_IO_H__ */
