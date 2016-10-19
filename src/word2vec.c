//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include "common.h"
#include "train.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>


/////////////
// Methods //
/////////////

/**
 * Print usage message and exit
 *
 * @param a_ret - exit code for the program
 *
 * @return \c void
 */
static void usage(int a_ret) {
  printf("WORD VECTOR estimation toolkit v 0.1c\n\n");
  printf("OPTIONS:\n");
  printf("Parameters for training:\n");
  printf("-train <file>\n");
  printf("\tUse text data from <file> to train the model\n");
  printf("-output <file>\n");
  printf("\tUse <file> to save the resulting word vectors / word clusters\n");
  printf("-size <int>\n");
  printf("\tSet size of word vectors; default is 100\n");
  printf("-window <int>\n");
  printf("\tSet max skip length between words; default is 5\n");
  printf("-sample <float>\n");
  printf("\tSet threshold for occurrence of words. Those that appear with higher frequency in the\n");
  printf("\ttraining data will be randomly down-sampled; default is 1e-3, useful range is (0, 1e-5)\n");
  printf("-hs <int>\n");
  printf("\tUse Hierarchical Softmax; default is 0 (not used)\n");
  printf("-negative <int>\n");
  printf("\tNumber of negative examples; default is 5, common values are 3 - 10 (0 = not used)\n");
  printf("-threads <int>\n");
  printf("\tUse <int> threads (default 12)\n");
  printf("-iter <int>\n");
  printf("\tRun more training iterations (default 5)\n");
  printf("-min-count <int>\n");
  printf("\tThis will discard words that appear less than <int> times; default is 5\n");
  printf("-alpha <float>\n");
  printf("\tSet the starting learning rate; default is 0.025 for skip-gram and 0.05 for CBOW\n");
  printf("-debug <int>\n");
  printf("\tSet the debug mode (default = 2 = more info during training)\n");
  printf("-binary <int>\n");
  printf("\tSave the resulting vectors in binary mode; default is 0 (off)\n");
  printf("-cbow <int>\n");
  printf("\tUse the continuous bag of words model; default is 1 (use 0 for skip-gram model)\n");
  printf("-multitask <int>\n");
  printf("\tTrain multi-task embeddings (argument should specify the number of additional tasks\n");
  printf("-task-specific <int>\n");
  printf("\tTrain task-specific embeddings; argument should specify the number of classes;\n");
  printf("\t(last field on each line of the input file should contain a numeric label `< int'\n");
  printf("\tor `_' if the label is unknown and the instance should be skipped)\n");
  printf("-least-sq <int>\n");
  printf("\tMap generic word2vec vectors to task-specific embeddings using least squares\n");
  printf("\nExamples:\n");
  printf("./word2vec -train data.txt -output vec.txt -size 200 -window 5 -sample 1e-4 -negative 5 -hs 0 -binary 0 -cbow 1 -iter 3\n\n");
  exit(a_ret);
}

int main(int argc, char **argv) {
  opt_t opt;
  reset_opt(&opt);

  int i;
  for (i = 1; i < (argc - 1); ++i) {
    if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
      usage(0);
    } else if (strcmp(argv[i], "-size") == 0) {
      opt.m_layer1_size = atoi(argv[++i]);
    } else if (strcmp(argv[i], "-train") == 0) {
      strcpy(opt.m_train_file, argv[++i]);
    } else if (strcmp(argv[i], "-debug") == 0) {
      opt.m_debug_mode = atoi(argv[++i]);
    } else if (strcmp(argv[i], "-binary") == 0) {
      opt.m_binary = atoi(argv[++i]);
    } else if (strcmp(argv[i], "-cbow") == 0) {
      if ((opt.m_cbow = atoi(argv[++i])))
        opt.m_alpha = 0.05;
    } else if (strcmp(argv[i], "-alpha") == 0) {
      opt.m_alpha = atof(argv[++i]);
    } else if (strcmp(argv[i], "-output") == 0) {
      strcpy(opt.m_output_file, argv[++i]);
    } else if (strcmp(argv[i], "-window") == 0) {
      opt.m_window = atoi(argv[++i]);
    } else if (strcmp(argv[i], "-sample") == 0) {
      opt.m_sample = atof(argv[++i]);
    } else if (strcmp(argv[i], "-hs") == 0) {
      opt.m_hs = atoi(argv[++i]);
    } else if (strcmp(argv[i], "-negative") == 0) {
      opt.m_negative = atoi(argv[++i]);
    } else if (strcmp(argv[i], "-threads") == 0) {
      opt.m_num_threads = atoi(argv[++i]);
    } else if (strcmp(argv[i], "-iter") == 0) {
      opt.m_iter = atoi(argv[++i]);
    } else if (strcmp(argv[i], "-min-count") == 0) {
      opt.m_min_count = atoi(argv[++i]);
    } else if (strcmp(argv[i], "-multitask") == 0) {
      opt.m_multitask = atoi(argv[++i]);
    } else if (strcmp(argv[i], "-task-specific") == 0) {
      opt.m_task_specific = atoi(argv[++i]);
    } else if (strcmp(argv[i], "-least-sq") == 0) {
      opt.m_least_sq = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--") == 0) {
      ++i;
      break;
    } else if ((*argv[i]) != '-') {
      break;
    } else {
      fprintf(stderr, "Unrecognized option: '%s'\n", argv[i]);
      usage(1);
    }
  }

  if (i != argc) {
    i = argc - 1;
    if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
      usage(0);
    }
    fprintf(stderr,
            "Missing argument for option: '%s'."
            "  Type --help to see usage.\n", argv[i]);
    exit(2);
  }

  if (opt.m_train_file[0] == 0) {
    fprintf(stderr,
            "No training file specified.  Type --help to see usage.\n");
    exit(3);
  }

  if (opt.m_least_sq)
    opt.m_task_specific = 1;

  if (opt.m_multitask && opt.m_task_specific) {
    fprintf(stderr,
            "Options -multitask and -task-specific/-least-sq are mutually exclusive."
            "  Type --help to see usage.\n");
    exit(4);
  }
  train_model(&opt);
  return 0;
}
