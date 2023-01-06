/*
 *
 * chCommandLine.h
 *
 * Dead simple command line parsing utility functions.
 *
 * Copyright (c) 2011-2012, Archaea Software, LLC.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in
 *    the documentation and/or other materials provided with the
 *    distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 */

// Example usage:

// int n_warmpups = 0;          //Default value
// chCommandLineGet<int>(&n_warmpups, "w", argc, argv);

// bool use_multi = chCommandLineGetBool("multi", argc, argv);

#ifndef __CUDAHANDBOOK_COMMANDLINE__
#define __CUDAHANDBOOK_COMMANDLINE__

#include <stdlib.h>
#include <string.h>

static void chCommandLinePassback(int *p, char const *s) { *p = atoi(s); }

static void chCommandLinePassback(unsigned long *p, char const *s) {
  *p = strtoul(s, NULL, 0);
}

// these functions work depending on the currently set local
// (. or , as decimal seperator problem) so be careful when using them
// static void chCommandLinePassback(float *p, char const *s) {
//   *p = strtod(s, NULL);
// }

// static void chCommandLinePassback(double *p, char const *s) {
//   *p = strtod(s, NULL);
// }

static void chCommandLinePassback(char const **p, char const *s) { *p = s; }

//
// Passes back an integer or string or unsigned long (for size_t)
//
template <typename T>
static bool chCommandLineGet(T *p, const char *keyword, int argc,
                             char const *const *argv) {
  bool ret = false;
  for (int i = 1; i < argc; i++) {
    char const *s = argv[i];
    if (*s == '-') {
      s++;
      if (*s == '-') {
        s++;
      }
      if (!strcmp(s, keyword)) {
        if (++i <= argc) {
          chCommandLinePassback(p, argv[i]);
          ret = true;
        }
      }
    }
  }
  return ret;
}

//
// Pass back true if the keyword is passed as a command line parameter
//
static bool chCommandLineGetBool(const char *keyword, int argc,
                                 char const *const *argv) {
  bool ret = false;
  for (int i = 1; i < argc; i++) {
    char const *s = argv[i];
    if (*s == '-') {
      s++;
      if (*s == '-') {
        s++;
      }
      if (!strcmp(s, keyword)) {
        return true;
      }
    }
  }
  return ret;
}

#endif