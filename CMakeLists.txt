CMAKE_MINIMUM_REQUIRED(VERSION 2.8.11)

PROJECT(word2vec)
SET(word2vec_VERSION_MAJOR 0)
SET(word2vec_VERSION_MINOR 1)

FIND_PACKAGE (Threads)

ADD_DEFINITIONS(-Wall -Wextra -fopenmp  -funroll-loops #
		      -march=native -funroll-loops -Ofast)

# specify output directory
SET(EXECUTABLE_OUTPUT_PATH bin)
SET(W2V_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src
  CACHE FILEPATH "Default directory containing word2vec source files")

# define targets

## word2vec
FILE(GLOB W2V_SOURCES
  "${W2V_SRC_DIR}/*.h"
  "${W2V_SRC_DIR}/*.c"
  )
ADD_EXECUTABLE(word2vec ${W2V_SOURCES})
TARGET_COMPILE_OPTIONS(word2vec PRIVATE -pthread -O3 -march=native)
TARGET_INCLUDE_DIRECTORIES(word2vec PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
TARGET_LINK_LIBRARIES(word2vec m pthread)
