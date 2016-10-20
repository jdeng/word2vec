//////////////
// Includes //
//////////////
#include "train.h"
#include "vocab.h"
#include "w2vio.h"

#include <math.h>
#include <pthread.h>
#include <stddef.h>
#include <stdio.h>
#include <time.h>

/////////////
// Structs //
/////////////
typedef struct {
  real *m_syn0;
  real *m_syn1;
  real *m_syn1neg;
} nnet_t;

typedef struct {
  clock_t m_start;
  size_t m_file_size;
  real m_alpha;
  real m_starting_alpha;
  long m_thread_id;
  const opt_t *m_w2v_opts;
  const vocab_t *m_vocab;
  nnet_t *m_nnet;
  real *m_exp_table;
  int *m_ugram_table;
} thread_opts_t;

///////////////
// Constants //
///////////////

/////////////
// Methods //
/////////////
static void reset_nnet(nnet_t *a_nnet) {
  a_nnet->m_syn0 = NULL;
  a_nnet->m_syn1 = NULL;
  a_nnet->m_syn1neg = NULL;
}

static void free_nnet(nnet_t *a_nnet) {
  free(a_nnet->m_syn0);
  free(a_nnet->m_syn1);
  free(a_nnet->m_syn1neg);
  reset_nnet(a_nnet);
}

static real *init_exp_table(void) {
  real *exp_table = (real *) malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
  int i;
  for (i = 0; i < EXP_TABLE_SIZE; ++i) {
    exp_table[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
    exp_table[i] = exp_table[i] / (exp_table[i] + 1); // Precompute f(x) = x / (x + 1)
  }
  return exp_table;
}

static void init_net(nnet_t *a_nnet, vocab_t *a_vocab, const opt_t *a_opts) {
  long long a, b;
  unsigned long long next_random = 1;
  long long vocab_size = a_vocab->m_vocab_size;
  long long layer1_size = a_opts->m_layer1_size;

  reset_nnet(a_nnet);
  fprintf(stderr, "nnet reset\n");

  a = posix_memalign((void **) &a_nnet->m_syn0, 128,
                     (long long) vocab_size
                     * layer1_size * sizeof(real));

  if (a_nnet->m_syn0 == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(1);
  }
  if (a_opts->m_hs) {
    a = posix_memalign((void **)&a_nnet->m_syn1, 128,
                       (long long) vocab_size
                       * layer1_size * sizeof(real));
    if (a_nnet->m_syn1 == NULL) {
      fprintf(stderr, "Memory allocation failed\n");
      exit(1);
    }
    for (a = 0; a < vocab_size; a++)
      for (b = 0; b < layer1_size; b++)
        a_nnet->m_syn1[a * layer1_size + b] = 0;
  }
  if (a_opts->m_negative > 0) {
    a = posix_memalign((void **)&a_nnet->m_syn1neg, 128,
                       (long long)vocab_size
                       * layer1_size * sizeof(real));
    if (a_nnet->m_syn1neg == NULL) {
      fprintf(stderr, "Memory allocation failed\n");
      exit(1);
    }
    for (a = 0; a < vocab_size; a++)
      for (b = 0; b < layer1_size; b++)
        a_nnet->m_syn1neg[a * layer1_size + b] = 0;
  }

  for (a = 0; a < vocab_size; a++) {
    for (b = 0; b < layer1_size; b++) {
      next_random = next_random * (unsigned long long)25214903917 + 11;
      a_nnet->m_syn0[a * layer1_size + b] = (((next_random & 0xFFFF)
                                              / (real)65536) - 0.5) / layer1_size;
    }
  }
}

static void *train_model_thread(void *a_opts) {
  thread_opts_t *thread_opts = (thread_opts_t *) a_opts;
  long long file_size = thread_opts->m_file_size;
  nnet_t *nnet = thread_opts->m_nnet;
  const real *exp_table = thread_opts->m_exp_table;
  const int *table = thread_opts->m_ugram_table;
  const vw_t *vocab = thread_opts->m_vocab->m_vocab;
  const int *vocab_hash = thread_opts->m_vocab->m_vocab_hash;
  long long vocab_size = thread_opts->m_vocab->m_vocab_size;
  long long train_words = thread_opts->m_vocab->m_train_words;

  const opt_t *w2v_opts = thread_opts->m_w2v_opts;
  long long layer1_size = w2v_opts->m_layer1_size;
  long long num_threads = w2v_opts->m_num_threads;
  real sample = w2v_opts->m_sample;
  const int window = w2v_opts->m_window;

  long long a, b, d, cw, word, last_word, sentence_length = 0, sentence_position = 0;
  long long word_count_actual = 0;
  long long word_count = 0, last_word_count = 0, sen[MAX_SENTENCE_LENGTH + 1];
  long long l1, l2, c, target, label;
  long long local_iter = w2v_opts->m_iter;
  unsigned long long next_random = (long long) thread_opts->m_thread_id;
  real f, g;
  clock_t now;
  real *neu1 = (real *)calloc(layer1_size, sizeof(real));
  real *neu1e = (real *)calloc(layer1_size, sizeof(real));
  FILE *fi = fopen(w2v_opts->m_train_file, "rb");
  fseek(fi, file_size / (long long) num_threads * (long long) next_random, SEEK_SET);
  while (1) {
    if (word_count - last_word_count > 10000) {
      word_count_actual += word_count - last_word_count;
      last_word_count = word_count;
      if ((w2v_opts->m_debug_mode > 1)) {
        now = clock();
        fprintf(stderr,
                "%cAlpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ", 13,
                thread_opts->m_alpha,
                word_count_actual / (real)(w2v_opts->m_iter * train_words + 1) * 100,
                word_count_actual / ((real)(now - thread_opts->m_start + 1)
                                     / (real)CLOCKS_PER_SEC * 1000));
        fflush(stderr);
      }
      thread_opts->m_alpha = thread_opts->m_starting_alpha              \
                             * (1 - word_count_actual /                 \
                                (real)(w2v_opts->m_iter * train_words + 1));
      if (w2v_opts->m_alpha < thread_opts->m_starting_alpha * 0.0001)
        thread_opts->m_alpha = thread_opts->m_starting_alpha * 0.0001;
    }

    if (sentence_length == 0) {
      while (1) {
        word = ReadWordIndex(fi, vocab, vocab_hash);
        if (feof(fi))
          break;

        if (word == -1)
          continue;
        ++word_count;
        if (word == 0) break;
        // The subsampling randomly discards frequent words while keeping the ranking same
        if (sample > 0) {
          real ran = (sqrt(vocab[word].cn /
                           (sample * train_words)) + 1)
                     * (sample * train_words) / vocab[word].cn;
          next_random = next_random * (unsigned long long)25214903917 + 11;
          if (ran < (next_random & 0xFFFF) / (real)65536) continue;
        }
        sen[sentence_length] = word;
        ++sentence_length;
        if (sentence_length >= MAX_SENTENCE_LENGTH) break;
      }
      sentence_position = 0;
    }
    if (feof(fi) || (word_count > train_words / num_threads)) {
      word_count_actual += word_count - last_word_count;
      local_iter--;
      if (local_iter == 0) break;
      word_count = 0;
      last_word_count = 0;
      sentence_length = 0;
      fseek(fi, file_size / (long long)num_threads * (long long)next_random, SEEK_SET);
      continue;
    }
    word = sen[sentence_position];
    if (word == -1)
      continue;

    for (c = 0; c < layer1_size; c++)
      neu1[c] = 0;

    for (c = 0; c < layer1_size; c++)
      neu1e[c] = 0;

    next_random = next_random * (unsigned long long)25214903917 + 11;
    b = next_random % window;
    if (w2v_opts->m_cbow) {  //train the cbow architecture
      // in -> hidden
      cw = 0;
      for (a = b; a < window * 2 + 1 - b; a++)
        if (a != window) {
          c = sentence_position - window + a;
          if (c < 0)
            continue;

          if (c >= sentence_length)
            continue;

          last_word = sen[c];
          if (last_word == -1)
            continue;

          for (c = 0; c < layer1_size; c++) {
            neu1[c] += nnet->m_syn0[c + last_word * layer1_size];
            /* fprintf(stderr, "neu1[%d] += nnet->m_syn0[%d] (%f)", */
            /*        c, c + last_word * layer1_size, */
            /*        nnet->m_syn0[c + last_word * layer1_size]); */
          }
          ++cw;
        }
      if (cw) {
        for (c = 0; c < layer1_size; c++)
          neu1[c] /= cw;

        if (w2v_opts->m_hs) {
          for (d = 0; d < vocab[word].codelen; d++) {
            f = 0;
            l2 = vocab[word].point[d] * layer1_size;
            // Propagate hidden -> output
            for (c = 0; c < layer1_size; c++)
              f += neu1[c] * nnet->m_syn1[c + l2];

            if (f <= -MAX_EXP || f >= MAX_EXP)
              continue;
            else
              f = exp_table[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
            // 'g' is the gradient multiplied by the learning rate
            g = (1 - vocab[word].code[d] - f) * w2v_opts->m_alpha;
            // Propagate errors output -> hidden
            for (c = 0; c < layer1_size; c++)
              neu1e[c] += g * nnet->m_syn1[c + l2];
            // Learn weights hidden -> output
            for (c = 0; c < layer1_size; c++)
              nnet->m_syn1[c + l2] += g * neu1[c];
          }
        }
        // NEGATIVE SAMPLING
        if (w2v_opts->m_negative > 0)
          for (d = 0; d < w2v_opts->m_negative + 1; d++) {
            if (d == 0) {
              target = word;
              label = 1;
            } else {
              next_random = next_random * (unsigned long long)25214903917 + 11;
              target = table[(next_random >> 16) % TABLE_SIZE];
              if (target == 0) target = next_random % (vocab_size - 1) + 1;
              if (target == word) continue;
              label = 0;
            }
            l2 = target * layer1_size;
            f = 0;
            for (c = 0; c < layer1_size; c++) f += neu1[c] * nnet->m_syn1neg[c + l2];
            if (f > MAX_EXP) g = (label - 1) * w2v_opts->m_alpha;
            else if (f < -MAX_EXP) g = (label - 0) * w2v_opts->m_alpha;
            else g = (label - exp_table[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * w2v_opts->m_alpha;
            for (c = 0; c < layer1_size; c++) neu1e[c] += g * nnet->m_syn1neg[c + l2];
            for (c = 0; c < layer1_size; c++) nnet->m_syn1neg[c + l2] += g * neu1[c];
          }
        // hidden -> in
        for (a = b; a < window * 2 + 1 - b; a++) {
          if (a != window) {
            c = sentence_position - window + a;
            if (c < 0) continue;
            if (c >= sentence_length) continue;
            last_word = sen[c];
            if (last_word == -1) continue;
            for (c = 0; c < layer1_size; c++)
              nnet->m_syn0[c + last_word * layer1_size] += neu1e[c];
          }
        }
      }
    } else {  //train skip-gram
      for (a = b; a < window * 2 + 1 - b; a++)
        if (a != window) {
          c = sentence_position - window + a;
          if (c < 0) continue;
          if (c >= sentence_length) continue;
          last_word = sen[c];
          if (last_word == -1) continue;
          l1 = last_word * layer1_size;
          for (c = 0; c < layer1_size; c++)
            neu1e[c] = 0;
          // HIERARCHICAL SOFTMAX
          if (w2v_opts->m_hs) for (d = 0; d < vocab[word].codelen; d++) {
              f = 0;
              l2 = vocab[word].point[d] * layer1_size;
              // Propagate hidden -> output
              for (c = 0; c < layer1_size; c++) f += nnet->m_syn0[c + l1] * nnet->m_syn1[c + l2];
              if (f <= -MAX_EXP) continue;
              else if (f >= MAX_EXP) continue;
              else f = exp_table[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
              // 'g' is the gradient multiplied by the learning rate
              g = (1 - vocab[word].code[d] - f) * w2v_opts->m_alpha;
              // Propagate errors output -> hidden
              for (c = 0; c < layer1_size; c++) neu1e[c] += g * nnet->m_syn1[c + l2];
              // Learn weights hidden -> output
              for (c = 0; c < layer1_size; c++) nnet->m_syn1[c + l2] += g * nnet->m_syn0[c + l1];
            }
          // NEGATIVE SAMPLING
          if (w2v_opts->m_negative > 0)
            for (d = 0; d < w2v_opts->m_negative + 1; ++d) {
              if (d == 0) {
                target = word;
                label = 1;
              } else {
                next_random = next_random * (unsigned long long)25214903917 + 11;
                target = table[(next_random >> 16) % TABLE_SIZE];
                if (target == 0) target = next_random % (vocab_size - 1) + 1;
                if (target == word) continue;
                label = 0;
              }
              l2 = target * layer1_size;
              f = 0;
              for (c = 0; c < layer1_size; c++)
                f += nnet->m_syn0[c + l1] * nnet->m_syn1neg[c + l2];

              if (f > MAX_EXP)
                g = (label - 1) * w2v_opts->m_alpha;
              else if (f < -MAX_EXP)
                g = (label - 0) * w2v_opts->m_alpha;
              else
                g = (label - exp_table[(int)((f + MAX_EXP)
                                            * (EXP_TABLE_SIZE / MAX_EXP / 2))])
                                            * w2v_opts->m_alpha;

              for (c = 0; c < layer1_size; c++)
                neu1e[c] += g * nnet->m_syn1neg[c + l2];

              for (c = 0; c < layer1_size; c++)
                nnet->m_syn1neg[c + l2] += g * nnet->m_syn0[c + l1];
            }
          // Learn weights input -> hidden
          for (c = 0; c < layer1_size; c++) nnet->m_syn0[c + l1] += neu1e[c];
        }
    }
    sentence_position++;
    if (sentence_position >= sentence_length) {
      sentence_length = 0;
      continue;
    }
  }
  fclose(fi);
  free(neu1);
  free(neu1e);
  pthread_exit(NULL);
}

static void save_embeddings(const opt_t *a_opts, const vocab_t *a_vocab,
                            const nnet_t *a_nnet) {
  FILE *fo = a_opts->m_output_file[0]? \
             fopen(a_opts->m_output_file, "wb"): stdout;

  long long vocab_size = a_vocab->m_vocab_size;
  const vw_t *vocab = a_vocab->m_vocab;

  // Save the word vectors
  fprintf(fo, "%lld %lld\n", vocab_size, a_opts->m_layer1_size);
  long a, b;
  for (a = 0; a < vocab_size; ++a) {
    fprintf(fo, "%s ", vocab[a].word);
    if (a_opts->m_binary)
      for (b = 0; b < a_opts->m_layer1_size; ++b)
        fwrite(&a_nnet->m_syn0[a * a_opts->m_layer1_size + b], sizeof(real), 1, fo);
    else
      for (b = 0; b < a_opts->m_layer1_size; ++b)
        fprintf(fo, "%lf ", a_nnet->m_syn0[a * a_opts->m_layer1_size + b]);

    fprintf(fo, "\n");
  }

  if (a_opts->m_output_file[0])
    fclose(fo);
}

void train_model(opt_t *a_opts) {
  fprintf(stderr, "Starting training using file '%s'\n",
          a_opts->m_train_file);

  /* initialize vocabulary and exp table */
  vocab_t vocab;
  init_vocab(&vocab);
  size_t file_size = learn_vocab_from_trainfile(&vocab, a_opts);

  real *exp_table = init_exp_table();

  nnet_t nnet;
  init_net(&nnet, &vocab, a_opts);

  int *ugram_table = NULL;
  if (a_opts->m_negative > 0)
    ugram_table = init_unigram_table(&vocab);

  thread_opts_t thread_opts = {clock(), file_size,
                               a_opts->m_alpha, a_opts->m_alpha,
                               0, a_opts, &vocab, &nnet,
                               exp_table, ugram_table};

  pthread_t *pt = (pthread_t *) malloc(a_opts->m_num_threads
                                       * sizeof(pthread_t));
  long a;
  for (a = 0; a < a_opts->m_num_threads; ++a) {
    thread_opts.m_thread_id = a;
    pthread_create(&pt[a], NULL, train_model_thread,
                   (void *) &thread_opts);
  }
  for (a = 0; a < a_opts->m_num_threads; a++)
    pthread_join(pt[a], NULL);

  save_embeddings(a_opts, &vocab, &nnet);

  free_nnet(&nnet);
  free(ugram_table);
  free(exp_table);
  free_vocab(&vocab);
}
