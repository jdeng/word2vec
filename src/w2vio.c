//////////////
// Includes //
//////////////
#include "w2vio.h"

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
int ReadWordIndex(FILE *fin) {
  char word[MAX_STRING];
  ReadWord(word, fin);
  if (feof(fin)) return -1;
  return SearchVocab(word);
}

static void process_line_w2v() {
  ++(*train_words);

  if ((a_opts->m_debug_mode > 1) && ((*train_words) % 100000 == 0)) {
    fprintf(stderr, "%lldK%c", train_words / 1000, 13);
    fflush(stderr);
  }

  i = SearchVocab(word);
  if (i == -1) {
    a = AddWordToVocab(word);
    vocab[a].cn = 1;
  } else {
    ++vocab[i].cn;
  }

  if (vocab_size > VOCAB_HASH_SIZE * 0.7)
    ReduceVocab();
}

void learn_vocab_from_trainfile(vocab_t *a_vocab, const opt_t *a_opts) {
  FILE *fin = fopen(a_opts->train_file, "rb");
  if (fin == NULL) {
    fprintf(stderr, "ERROR: training data file not found!\n");
    exit(EXIT_FAILURE);
  }

  for (a = 0; a < VOCAB_HASH_SIZE; a++) {
    a_vocab->m_vocab_hash[a] = -1;
  }

  /* size_t len = 0; */
  /* ssize_t read; */
  /* char *line = NULL; */
  /* char word[MAX_STRING]; */
  /* long long vocab_size = 0, train_words = 0; */
  /* void (*process_line)(const char *, ssize_t, int *) = NULL; */

  /* if (a_opts->m_multitask) */
  /*   process_line = process_line_multitask; */
  /* else if (a_opts->m_task_specific) */
  /*   process_line = process_line_task_specific; */
  /* else */
  /*   process_line = process_line_w2v; */

  /* while ((read = getline(&line, &len, fin)) != -1) { */
  /*   if ((a_opts->m_debug_mode > 1) && (train_words % 100000 == 0)) { */
  /*     fprintf(stderr, "%lldK%c", train_words / 1000, 13); */
  /*     fflush(stderr); */
  /*   } */
  /*   process_line(line, read); */
  /* } */
  /* free(line); */

  /* if (ferror(fin)) { */
  /*   fprintf(stderr, "ERROR: reading input file\n"); */
  /*   exit(EXIT_FAILURE); */
  /* } */
  /* SortVocab(); */
  /* if (a_opts->m_debug_mode > 0) { */
  /*   fprintf(stderr, "Vocab size: %lld\n", vocab_size); */
  /*   fprintf(stderr, "Words in train file: %lld\n", train_words); */
  /* } */
  /* long long file_size = ftell(fin); */
  /* fclose(fin); */
  /* return file_size; */
}
