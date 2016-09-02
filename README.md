word2vec
========

Word2Vec in C++ 11

See main.cc for building instructions and usage. (NOTE: openmp is used in the newest version and thus g++ is required for multithreading)

Results with OMP_NUM_THREADS=8: (save model is understandably slow as it stores text)

    loadvocab: 1.9952 seconds    
    train: 33.5145 seconds
    save model: 4.7554 seconds
  
  
Machine configuration

    jackdeng-mac:word2vec jack$ sysctl -n machdep.cpu.brand_string
    Intel(R) Core(TM) i7-4850HQ CPU @ 2.30GHz
    
