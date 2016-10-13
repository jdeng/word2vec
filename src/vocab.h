#ifndef __WORD2VEC_VOCAB_H__
#define __WORD2VEC_VOCAB_H__

//////////////
// Includes //
//////////////
#include <stddef.h>                   /* NULL */
#include <stdlib.h>                   /* malloc(), free() */

///////////////
// Constants //
///////////////
const int MAX_CODE_LENGTH = 40;
const int MAX_SENTENCE_LENGTH = 1000;
const int VOCAB_HASH_SIZE = 30000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary

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

real *InitUnigramTable();

/**
 * Look up a word in the vocabulary.
 *
 * @param word - word to search for
 * @type char *
 *
 * @return \c int - position of a word in the vocabulary or -1 if the
 *   word is not found
 */
int SearchVocab(char *word);

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
 * @param a_vocab - vocabulary innstance
 * @type vocab_t *
 *
 * @return \c void
 */
void init_vocab(vocab_t *a_vocab) {
  a_vocab->m_vocab_size = 0;
  a_vocab->m_vocab = NULL;
  a_vocab->m_vocab_hash = (int *) calloc(VOCAB_HASH_SIZE, sizeof(int));
}

/**
 * Free memory occupied by vocabulary.
 *
 * @param a_vocab - vocabulary innstance
 * @type vocab_t *
 *
 * @return \c void
 */
void free_vocab(vocab_t *a_vocab) {
  a_vocab->m_vocab_size = 0;
  free(a_vocab->m_vocab);
  free(a_vocab->m_vocab_hash);
}

#endif  /* ifndef __WORD2VEC_VOCAB_H__ */
