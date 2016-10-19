//////////////
// Includes //
//////////////
#include "vocab.h"

#include <math.h>   /* pow() */
#include <string.h> /* strcmp() */
#include <stdio.h>  /* fprintf() */

///////////////
// Constants //
///////////////
const int TABLE_SIZE = 1e8;
const int MAX_CODE_LENGTH = 40;
const int MAX_SENTENCE_LENGTH = 1000;
const int VOCAB_HASH_SIZE = 30000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary
const char EOS[] = "</s>";

/////////////
// Methods //
/////////////
int *InitUnigramTable(vocab_t *a_vocab) {
  vw_t *vocab = a_vocab->m_vocab;
  long long vocab_size = a_vocab->m_vocab_size;
  if (vocab_size == 0)
    return NULL;

  long long train_words_pow = 0;
  real d1, power = 0.75;
  int *table = (int *) malloc(TABLE_SIZE * sizeof(int));
  long long a;
  for (a = 0; a < vocab_size; a++)
    train_words_pow += pow(vocab[a].cn, power);

  int i = 0;
  d1 = pow(vocab[i].cn, power) / (real) train_words_pow;
  for (a = 0; a < TABLE_SIZE; ++a) {
    table[a] = i;
    if (a / (real) TABLE_SIZE > d1) {
      ++i;
      d1 += pow(vocab[i].cn, power) / (real) train_words_pow;
    }
    if (i >= vocab_size)
      i = vocab_size - 1;
  }
  return table;
}

// Create binary Huffman tree using the word counts
// Frequent words will have short uniqe binary codes
void create_binary_tree(vocab_t *a_vocab) {
  char code[MAX_CODE_LENGTH];
  long long a, b, i, min1i, min2i, pos1, pos2, point[MAX_CODE_LENGTH];

  vw_t *vocab = a_vocab->m_vocab;
  long long vocab_size = a_vocab->m_vocab_size;
  long long *count = (long long *) calloc(vocab_size * 2 + 1, sizeof(long long));
  long long *binary = (long long *) calloc(vocab_size * 2 + 1, sizeof(long long));
  long long *parent_node = (long long *) calloc(vocab_size * 2 + 1, sizeof(long long));

  for (a = 0; a < vocab_size; a++)
    count[a] = vocab[a].cn;

  for (a = vocab_size; a < vocab_size * 2; a++)
    count[a] = 1e15;

  pos1 = vocab_size - 1;
  pos2 = vocab_size;
  // Following algorithm constructs the Huffman tree by adding one node at a time
  for (a = 0; a < vocab_size - 1; ++a) {
    // First, find two smallest nodes 'min1, min2'
    if (pos1 >= 0) {
      if (count[pos1] < count[pos2]) {
        min1i = pos1;
        --pos1;
      } else {
        min1i = pos2;
        ++pos2;
      }
    } else {
      min1i = pos2;
      ++pos2;
    }
    if (pos1 >= 0) {
      if (count[pos1] < count[pos2]) {
        min2i = pos1;
        --pos1;
      } else {
        min2i = pos2;
        ++pos2;
      }
    } else {
      min2i = pos2;
      ++pos2;
    }
    count[vocab_size + a] = count[min1i] + count[min2i];
    parent_node[min1i] = vocab_size + a;
    parent_node[min2i] = vocab_size + a;
    binary[min2i] = 1;
  }
  // Now assign binary code to each vocabulary word
  for (a = 0; a < vocab_size; ++a) {
    b = a;
    i = 0;
    while (1) {
      code[i] = binary[b];
      point[i] = b;
      ++i;
      b = parent_node[b];
      if (b == vocab_size * 2 - 2)
        break;
    }
    vocab[a].codelen = i;
    vocab[a].point[0] = vocab_size - 2;
    for (b = 0; b < i; b++) {
      vocab[a].code[i - b - 1] = code[b];
      vocab[a].point[i - b] = point[b] - vocab_size;
    }
  }
  free(count);
  free(binary);
  free(parent_node);
}

// Returns hash value of a word
static int GetWordHash(const char *word) {
  unsigned long long a, hash = 0;
  for (a = 0; a < strlen(word); ++a)
    hash = hash * 257 + word[a];

  return hash % VOCAB_HASH_SIZE;
}

int SearchVocab(const char *a_word, const vw_t *a_vocab, const int *a_vocab_hash) {
  unsigned int hash = GetWordHash(a_word);
  while (1) {
    if (a_vocab_hash[hash] == -1)
      return -1;

    if (!strcmp(a_word, a_vocab[a_vocab_hash[hash]].word))
      return a_vocab_hash[hash];

    hash = (hash + 1) % VOCAB_HASH_SIZE;
  }
  return -1;
}

// Adds a word to the vocabulary
int AddWordToVocab(vocab_t *a_vocab, const char *a_word) {
  long long vocab_size = a_vocab->m_vocab_size;
  int *vocab_hash = a_vocab->m_vocab_hash;
  vw_t *vocab = a_vocab->m_vocab;


  /* check if the word is already known */
  int i;
  if ((i = SearchVocab(a_word, vocab, vocab_hash)) >= 0) {
    ++vocab[i].cn;
    return vocab_size;
  }
  /* truncate word to the maximum acceptable length */
  unsigned int hash, length = strlen(a_word) + 1;
  if (length > MAX_STRING)
    length = MAX_STRING;

  /* Reallocate memory if needed */
  if (vocab_size + 2 >= a_vocab->m_max_vocab_size) {
    a_vocab->m_max_vocab_size += 1000;
    a_vocab->m_vocab = (vw_t *) realloc(a_vocab->m_vocab,
                                        a_vocab->m_max_vocab_size * sizeof(vw_t));
    vocab = a_vocab->m_vocab;
  }

  vocab[vocab_size].word = (char *) calloc(length, sizeof(char));
  strcpy(vocab[vocab_size].word, a_word);
  vocab[vocab_size].cn = 1;
  vocab_size = ++a_vocab->m_vocab_size;

  hash = GetWordHash(a_word);
  while (vocab_hash[hash] != -1)
    hash = (hash + 1) % VOCAB_HASH_SIZE;

  vocab_hash[hash] = vocab_size - 1;
  return vocab_size - 1;
}

// Used later for sorting by word counts
int VocabCompare(const void *a, const void *b) {
  return ((struct vocab_word *)b)->cn - ((struct vocab_word *)a)->cn;
}

// Sorts the vocabulary by frequency using word counts
int sort_vocab(vocab_t *a_vocab, const int a_min_count) {
  long long vocab_size = a_vocab->m_vocab_size;
  vw_t *vocab = a_vocab->m_vocab;
  int *vocab_hash = a_vocab->m_vocab_hash;

  int a, size;
  unsigned int hash;
  // Sort the vocabulary and keep </s> at the first position
  qsort(&vocab[1], vocab_size - 1,
        sizeof(struct vocab_word), VocabCompare);

  for (a = 0; a < VOCAB_HASH_SIZE; ++a) {
    vocab_hash[a] = -1;
  }
  size = vocab_size;
  int train_words = 0;
  for (a = 0; a < size; a++) {
    // Words occuring less than min_count times will be discarded from the vocab
    if ((vocab[a].cn < a_min_count) && (a != 0)) {
      vocab_size = --a_vocab->m_vocab_size;
      free(vocab[a].word);
    } else {
      // Hash will be re-computed, as after the sorting it is not actual
      hash = GetWordHash(vocab[a].word);
      while (vocab_hash[hash] != -1) {
        hash = (hash + 1) % VOCAB_HASH_SIZE;
      }
      vocab_hash[hash] = a;
      train_words += vocab[a].cn;
    }
  }
  vocab = (struct vocab_word *) realloc(vocab, (vocab_size + 1)
                                        * sizeof(struct vocab_word));
  // Allocate memory for the binary tree construction
  for (a = 0; a < vocab_size; ++a) {
    vocab[a].code = (char *) calloc(MAX_CODE_LENGTH, sizeof(char));
    vocab[a].point = (int *) calloc(MAX_CODE_LENGTH, sizeof(int));
  }
  return train_words;
}

// Reduces the vocabulary by removing infrequent tokens
void reduce_vocab(vocab_t *a_vocab, opt_t *a_opts) {
  long long vocab_size = a_vocab->m_vocab_size;
  vw_t *vocab = a_vocab->m_vocab;
  int *vocab_hash = a_vocab->m_vocab_hash;

  int a, b = 0;
  unsigned int hash;
  for (a = 0; a < vocab_size; a++) {
    if (vocab[a].cn > a_opts->m_min_reduce) {
      vocab[b].cn = vocab[a].cn;
      vocab[b].word = vocab[a].word;
      ++b;
    } else {
      free(vocab[a].word);
    }
  }
  for (a = 0; a < VOCAB_HASH_SIZE; a++) {
    vocab_hash[a] = -1;
  }
  a_vocab->m_vocab_size = b;
  vocab_size = a_vocab->m_vocab_size;
  for (a = 0; a < vocab_size; a++) {
    // Hash will be re-computed, as it is not actual
    hash = GetWordHash(vocab[a].word);
    while (vocab_hash[hash] != -1) {
      hash = (hash + 1) % VOCAB_HASH_SIZE;
    }
    vocab_hash[hash] = a;
  }
  ++a_opts->m_min_reduce;
}

void init_vocab(vocab_t *a_vocab) {
  a_vocab->m_vocab_size = 0; /**< number of actually stored elements */
  a_vocab->m_max_vocab_size = 0;  /**< pre-allocated  */
  a_vocab->m_vocab = NULL;
  a_vocab->m_vocab_hash = (int *) calloc(VOCAB_HASH_SIZE, sizeof(int));
}

void free_vocab(vocab_t *a_vocab) {
  free(a_vocab->m_vocab);
  free(a_vocab->m_vocab_hash);
  a_vocab->m_max_vocab_size = 0;
  a_vocab->m_vocab_size = 0;
}
