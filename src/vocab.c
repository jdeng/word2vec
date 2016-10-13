//////////////
// Includes //
//////////////
#include "vocab.h"

#include <string.h>  /* strcmp() */

///////////////
// Variables //
///////////////
static long long vocab_max_size = 1000, vocab_size = 0;
static int *vocab_hash;

/////////////
// Methods //
/////////////

real *InitUnigramTable() {
  int a, i;
  long long train_words_pow = 0;
  real d1, power = 0.75;
  table = (int *)malloc(table_size * sizeof(int));
  for (a = 0; a < vocab_size; a++) train_words_pow += pow(vocab[a].cn, power);
  i = 0;
  d1 = pow(vocab[i].cn, power) / (real)train_words_pow;
  for (a = 0; a < table_size; a++) {
    table[a] = i;
    if (a / (real)table_size > d1) {
      i++;
      d1 += pow(vocab[i].cn, power) / (real)train_words_pow;
    }
    if (i >= vocab_size) i = vocab_size - 1;
  }
  return table;
}

// Create binary Huffman tree using the word counts
// Frequent words will have short uniqe binary codes
void CreateBinaryTree(vocab_t *a_vocab) {
  char code[MAX_CODE_LENGTH];
  long long a, b, i, min1i, min2i, pos1, pos2, point[MAX_CODE_LENGTH];

  long long vocab_size = a_vocab->m_vocab_size;
  long long *count = (long long *) calloc(vocab_size * 2 + 1, sizeof(long long));
  long long *binary = (long long *) calloc(vocab_size * 2 + 1, sizeof(long long));
  long long *parent_node = (long long *) calloc(vocab_size * 2 + 1, sizeof(long long));

  for (a = 0; a < vocab_size; a++)
    count[a] = a_vocab[a].cn;

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
  for (a = 0; a < vocab_size; a++) {
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
    a_vocab[a].codelen = i;
    a_vocab[a].point[0] = vocab_size - 2;
    for (b = 0; b < i; b++) {
      a_vocab[a].code[i - b - 1] = code[b];
      a_vocab[a].point[i - b] = point[b] - vocab_size;
    }
  }
  free(count);
  free(binary);
  free(parent_node);
}

// Returns hash value of a word
static int GetWordHash(char *word) {
  unsigned long long a, hash = 0;
  for (a = 0; a < strlen(word); a++)
    hash = hash * 257 + word[a];

  return hash % VOCAB_HASH_SIZE;
}

int SearchVocab(char *word, int *vocab_hash) {
  unsigned int hash = GetWordHash(word);
  while (1) {
    if (vocab_hash[hash] == -1)
      return -1;
    if (!strcmp(word, vocab[vocab_hash[hash]].word))
      return vocab_hash[hash];

    hash = (hash + 1) % VOCAB_HASH_SIZE;
  }
  return -1;
}

// Adds a word to the vocabulary
int AddWordToVocab(char *word, int *vocab_hash) {
  unsigned int hash, length = strlen(word) + 1;
  if (length > MAX_STRING) length = MAX_STRING;
  vocab[vocab_size].word = (char *)calloc(length, sizeof(char));
  strcpy(vocab[vocab_size].word, word);
  vocab[vocab_size].cn = 0;
  vocab_size++;
  // Reallocate memory if needed
  if (vocab_size + 2 >= vocab_max_size) {
    vocab_max_size += 1000;
    vocab = (struct vocab_word *)realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
  }
  hash = GetWordHash(word);
  while (vocab_hash[hash] != -1) hash = (hash + 1) % VOCAB_HASH_SIZE;
  vocab_hash[hash] = vocab_size - 1;
  return vocab_size - 1;
}

// Used later for sorting by word counts
int VocabCompare(const void *a, const void *b) {
  return ((struct vocab_word *)b)->cn - ((struct vocab_word *)a)->cn;
}

// Sorts the vocabulary by frequency using word counts
void SortVocab(int *vocab_hash) {
  int a, size;
  unsigned int hash;
  // Sort the vocabulary and keep </s> at the first position
  qsort(&vocab[1], vocab_size - 1, sizeof(struct vocab_word), VocabCompare);
  for (a = 0; a < VOCAB_HASH_SIZE; a++) vocab_hash[a] = -1;
  size = vocab_size;
  train_words = 0;
  for (a = 0; a < size; a++) {
    // Words occuring less than min_count times will be discarded from the vocab
    if ((vocab[a].cn < min_count) && (a != 0)) {
      vocab_size--;
      free(vocab[a].word);
    } else {
      // Hash will be re-computed, as after the sorting it is not actual
      hash=GetWordHash(vocab[a].word);
      while (vocab_hash[hash] != -1) hash = (hash + 1) % VOCAB_HASH_SIZE;
      vocab_hash[hash] = a;
      train_words += vocab[a].cn;
    }
  }
  vocab = (struct vocab_word *)realloc(vocab, (vocab_size + 1) * sizeof(struct vocab_word));
  // Allocate memory for the binary tree construction
  for (a = 0; a < vocab_size; a++) {
    vocab[a].code = (char *)calloc(MAX_CODE_LENGTH, sizeof(char));
    vocab[a].point = (int *)calloc(MAX_CODE_LENGTH, sizeof(int));
  }
}

// Reduces the vocabulary by removing infrequent tokens
void ReduceVocab(int *vocab_hash) {
  int a, b = 0;
  unsigned int hash;
  for (a = 0; a < vocab_size; a++) {
    if (vocab[a].cn > min_reduce) {
      vocab[b].cn = vocab[a].cn;
      vocab[b].word = vocab[a].word;
      b++;
    } else {
      free(vocab[a].word);
    }
  }
  vocab_size = b;
  for (a = 0; a < VOCAB_HASH_SIZE; a++) {
    vocab_hash[a] = -1;
  }
  for (a = 0; a < vocab_size; a++) {
    // Hash will be re-computed, as it is not actual
    hash = GetWordHash(vocab[a].word);
    while (vocab_hash[hash] != -1) hash = (hash + 1) % VOCAB_HASH_SIZE;
    vocab_hash[hash] = a;
  }
  fflush(stdout);
  min_reduce++;
}
