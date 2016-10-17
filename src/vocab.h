#ifndef __WORD2VEC_VOCAB_H__
#define __WORD2VEC_VOCAB_H__

//////////////
// Includes //
//////////////
#include "common.h"                   /* real */

#include <stddef.h>                   /* NULL */
#include <stdlib.h>                   /* malloc(), free() */

////////////
// Macros //
////////////

///////////////
// Constants //
///////////////
extern const int TABLE_SIZE;
extern const int MAX_CODE_LENGTH;
extern const int MAX_SENTENCE_LENGTH;
extern const int VOCAB_HASH_SIZE;  // Maximum 30 * 0.7 = 21M words in the vocabulary

/////////////
// Structs //
/////////////
typedef struct vocab_word {
  long long cn;
  int *point;
  char *word, *code, codelen;
} vw_t;

typedef struct vocab {
  long long m_vocab_size;
  vw_t *m_vocab;
  int *m_vocab_hash;
} vocab_t;

/////////////
// Methods //
/////////////

/**
 * Initialize a unigram table.
 *
 * @param a_vocab - vocabulary with relevant information
 * @type vw_t *
 *
 * @return \c int * - pointer to the initialized table
 */
int *InitUnigramTable(vocab_t *a_vocab);

/**
 * Add a word to the vocabulary.
 *
 * @param a_vocab - vocabulary to add the word to
 * @type vocab_t *
 * @param a_word - word to be added
 * @type char *
 *
 * @return \c int - position of a word in the vocabulary
 */
int AddWordToVocab(vocab_t *a_vocab, char *a_word);

/**
 * Look up a word in the vocabulary.
 *
 * @param a_word - word to search for
 * @type char *
 * @param a_vocab - vocabulary to search in
 * @type vw_t *
 * @param a_vocab_hash - hash of word indices
 * @type int *
 *
 * @return \c int - position of a word in the vocabulary or -1 if the
 *   word is not found
 */
int SearchVocab(char *a_word, const vw_t *a_vocab, const int *a_vocab_hash);

/**
 * Create binary search tree for vocabulary.
 *
 * @param a_vocab - vocabulary innstance
 * @type vocab_t *
 *
 * @return \c void
 */
void CreateBinaryTree(vocab_t *a_vocab);

/**
 * Initialize vocabulary to an empty dictionary.
 *
 * @param a_vocab - vocabulary instance
 * @type vocab_t *
 *
 * @return \c void
 */
void init_vocab(vocab_t *a_vocab);

/**
 * Sort vocabulary.
 *
 * @param a_vocab - vocabulary to be sorted
 * @type vocab_t *
 * @param a_min_count - minimum required word frequency
 * @type const int
 *
 * @return \c int - number of words in the vocabulary
 */
int sort_vocab(vocab_t *a_vocab, const int a_min_count);

/**
 * Free memory occupied by vocabulary.
 *
 * @param a_vocab - vocabulary innstance
 * @type vocab_t *
 *
 * @return \c void
 */
void free_vocab(vocab_t *a_vocab);

#endif  /* ifndef __WORD2VEC_VOCAB_H__ */
