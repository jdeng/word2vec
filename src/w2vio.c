//////////////
// Includes //
//////////////
#include "common.h"
#include "w2vio.h"

#include <ctype.h>  /* isspace() */
#include <string.h>

/////////////
// Methods //
/////////////

// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
void ReadWord(char *word, FILE *fin) {
  int a = 0, ch;
  while (!feof(fin)) {
    ch = fgetc(fin);
    if (ch == 13) continue;
    if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
      if (a > 0) {
        if (ch == '\n') ungetc(ch, fin);
        break;
      }
      if (ch == '\n') {
        strcpy(word, (char *)"</s>");
        return;
      } else continue;
    }
    word[a] = ch;
    a++;
    if (a >= MAX_STRING - 1) a--;   // Truncate too long words
  }
  word[a] = 0;
}

// Reads a word and returns its index in the vocabulary
int ReadWordIndex(FILE *fin,
                  const vw_t *a_vocab, const int *a_vocab_hash) {
  char word[MAX_STRING];
  ReadWord(word, fin);
  if (feof(fin))
    return -1;

  return search_vocab(word, a_vocab, a_vocab_hash);
}

static int process_line_multitask(vocab_t *a_vocab, char *a_word,
                                  const char *a_line, ssize_t a_read) {
  int n_words = 0;
  return n_words;
}

static int process_line_task_specific(vocab_t *a_vocab, char *a_word,
                                      const char *a_line, ssize_t a_read) {
  int n_words = 0;
  return n_words;
}

static int process_line_w2v(vocab_t *a_vocab, char *a_word,
                            const char *a_line, ssize_t a_read) {
  int n_words = 0;

  ssize_t i, n_chars;
  for (i = 0, n_chars = 0; i < a_read; ++i, ++n_chars) {
    if (isspace(a_line[i])) {
      if (n_chars > 0) {
        a_word[n_chars] = 0;
        add_word2vocab(a_vocab, a_word);
      }
      n_chars = -1;
    } else if (n_chars >= MAX_STRING) {
      n_chars = MAX_STRING - 1;
      a_word[n_chars] = 0;
    } else {
      a_word[n_chars] = a_line[i];
    }
  }

  if (n_chars > 0)
    add_word2vocab(a_vocab, a_word);

  return n_words;
}

size_t learn_vocab_from_trainfile(vocab_t *a_vocab, opt_t *a_opts) {
  FILE *fin = fopen(a_opts->m_train_file, "rb");
  if (fin == NULL) {
    fprintf(stderr, "ERROR: training data file not found!\n");
    exit(EXIT_FAILURE);
  }

  int a;
  for (a = 0; a < VOCAB_HASH_SIZE; a++) {
    a_vocab->m_vocab_hash[a] = -1;
  }

  ssize_t read;
  char *line = NULL;
  size_t len = 0;
  char word[MAX_STRING];
  int (*process_line)(vocab_t *a_vocab, char *a_word,
                      const char *a_line, ssize_t a_read) = NULL;

  if (a_opts->m_multitask)
    process_line = process_line_multitask;
  else if (a_opts->m_task_specific)
    process_line = process_line_task_specific;
  else
    process_line = process_line_w2v;

  add_word2vocab(a_vocab, EOS);
  long long train_words = 1;
  while ((read = getline(&line, &len, fin)) != -1) {
    if ((a_opts->m_debug_mode > 1) && (train_words % 100000 == 0)) {
      fprintf(stderr, "%lldK%c", train_words / 1000, 13);
      fflush(stderr);
    }
    train_words += process_line(a_vocab, word, line, read);

    if (a_vocab->m_vocab_size > VOCAB_HASH_SIZE * 0.7)
      reduce_vocab(a_vocab, a_opts);
  }
  free(line);
  sort_vocab(a_vocab, a_opts->m_min_count);
  create_binary_tree(a_vocab);

  if (ferror(fin)) {
    fprintf(stderr, "ERROR: reading input file\n");
    exit(EXIT_FAILURE);
  }
  if (a_opts->m_debug_mode > 0) {
    fprintf(stderr, "Vocab size: %lld\n", a_vocab->m_vocab_size);
    fprintf(stderr, "Words in train file: %lld\n", train_words);
  }
  size_t file_size = ftell(fin);
  fclose(fin);
  return file_size;
}
