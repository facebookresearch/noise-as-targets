/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <limits.h>

struct data {
  int m;
  int n;
  int pathLength;
  int* icost;
  int* cost;
  int* mark;
  bool* colCover;
  bool* rowCover;
  int* path;
};

struct data* newData(int m, int n) {
  struct data* prob = (struct data*) malloc(sizeof(struct data));
  prob->m = m;
  prob->n = n;
  prob->pathLength = 0;
  prob->cost = (int*) malloc(m * n * sizeof(int));
  prob->icost = (int*) malloc(m * n * sizeof(int));
  prob->mark = (int*) malloc(m * n * sizeof(int));
  prob->colCover = (bool*) malloc(n * sizeof(bool));
  prob->rowCover = (bool*) malloc(m * sizeof(bool));
  prob->path = (int*) malloc((2 * m + 1) * 2 * sizeof(int));
  for (int i = 0; i < 2 * (2 * m + 1); i++) {
    prob->path[i] = 0;
  }
  for (int i = 0; i < m * n; i++) {
      prob->cost[i] = 0;
      prob->icost[i] = 0;
      prob->mark[i] = 0;
  }
  for (int i = 0; i < m; i++) {
    prob->rowCover[i] = false;
  }
  for (int j = 0; j < n; j++) {
    prob->colCover[j] = false;
  }
  return prob;
}

void freeData(struct data* prob) {
  free(prob->cost);
  free(prob->icost);
  free(prob->mark);
  free(prob->colCover);
  free(prob->rowCover);
  free(prob);
}

void initData(struct data* prob) {
  for (int i = 0; i < prob->m; i++) {
    for (int j = 0; j < prob->n; j++) {
      prob->cost[i * prob->n + j] = rand() % 100;
      prob->icost[i * prob->n + j] = prob->cost[i * prob->n + j];
      // prob->cost[i * prob->n + j] = i * j;
    }
  }
}

struct data* readData() {
  int m = 0;
  int n = 0;
  scanf("%d %d", &m, &n);
  struct data* prob = newData(m, n);
  for (int i = 0; i < prob->m; i++) {
    for (int j = 0; j < prob->n; j++) {
      int temp = 0;
      scanf("%d", &temp);
      prob->cost[i * prob->n + j] = temp;
      prob->icost[i * prob->n + j] = prob->cost[i * prob->n + j];
    }
  }
  return prob;
}

struct data* fromPointer(int m, int n, const int* c) {
  struct data* prob = newData(m, n);
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      prob->cost[i * n + j] = c[i * n + j];
      prob->icost[i * n + j] = c[i * n + j];
    }
  }
  return prob;
}

void printCost(struct data* prob) {
  for (int i = 0; i < prob->m; i++) {
    for (int j = 0; j < prob->n; j++) {
      printf("%9d ", prob->cost[i * prob->n + j]);
    }
    printf("\n");
  }
}

void printMark(struct data* prob) {
  for (int i = 0; i < prob->m; i++) {
    for (int j = 0; j < prob->n; j++) {
      printf("%1d ", prob->mark[i * prob->n + j]);
    }
    printf("\n");
  }
}

void clearCovers(struct data* prob) {
  for (int i = 0; i < prob->m; i++) {
    prob->rowCover[i] = false;
  }
  for (int j = 0; j < prob->n; j++) {
    prob->colCover[j] = false;
  }
}

void clearPrimes(struct data* prob) {
  for (int i = 0; i < prob->m; i++) {
    for (int j = 0; j < prob->n; j++) {
      if (prob->mark[i * prob->n + j] == 2) {
        prob->mark[i * prob->n + j] = 0;
      }
    }
  }
}

bool starInRow(struct data* prob, int r) {
  bool res = false;
  for (int j = 0; j < prob->n; j++) {
    if (prob->mark[r * prob->n + j] == 1) {
      res = true;
      break;
    }
  }
  return res;
}

int findStarInRow(struct data* prob, int r) {
  int res = -1;
  for (int j = 0; j < prob->n; j++) {
    if (prob->mark[r * prob->n + j] == 1) {
      res = j;
      break;
    }
  }
  return res;
}

int findStarInCol(struct data* prob, int c) {
  int res = -1;
  for (int i = 0; i < prob->m; i++) {
    if (prob->mark[i * prob->n + c] == 1) {
      res = i;
      break;
    }
  }
  return res;
}

int findPrimeInRow(struct data* prob, int r) {
  int res = -1;
  for (int j = 0; j < prob->n; j++) {
    if (prob->mark[r * prob->n + j] == 2) {
      res = j;
      break;
    }
  }
  return res;
}

void findUncoveredZero(struct data* prob, int* row, int* col) {
  (*row) = -1;
  (*col) = -1;
  bool done = false;
  int m = prob->m;
  int n = prob->n;
  int i = 0;
  int j = 0;
  for (i = 0; i < m; i++) {
    if (!prob->rowCover[i]) {
      int in = i * n;
      for (j = 0; j < n; j++) {
        if (prob->cost[in + j] == 0
            && !prob->colCover[j]) {
          (*row) = i;
          (*col) = j;
          done = true;
          break;
        }
      }
    }
    if (done) {
      break;
    }
  }
}

void augmentPath(struct data* prob) {
  int i = 0;
  int j = 0;
  for (int p = 0; p < prob->pathLength; p++) {
    i = prob->path[2 * p];
    j = prob->path[2 * p + 1];
    if (prob->mark[i * prob->n + j] == 1) {
      prob->mark[i * prob->n + j] = 0;
    } else {
      prob->mark[i * prob->n + j] = 1;
    }
  }
}

int minUncovered(struct data* prob) {
  int res = INT_MAX;
  for (int i = 0; i < prob->m; i++) {
    for (int j = 0; j < prob->n; j++) {
      if (!prob->rowCover[i]
          && !prob->colCover[j]
          && prob->cost[i * prob->n + j] < res) {
        res = prob->cost[i * prob->n + j];
      }
    }
  }
  return res;
}

void stepOne(struct data* prob) {
  int min = 0.0;
  for (int i = 0; i < prob->m; i++) {
    min = prob->cost[i * prob->n];
    for (int j = 0; j < prob->n; j++) {
      if (prob->cost[i * prob->n + j] < min) {
        min = prob->cost[i * prob->n + j];
      }
    }
    for (int j = 0; j < prob->n; j++) {
      prob->cost[i * prob->n + j] -= min;
    }
  }
}

void stepTwo(struct data* prob) {
  for (int i = 0; i < prob->m; i++) {
    for (int j = 0; j < prob->n; j++) {
      if (prob->cost[i * prob->n + j] == 0
          && !prob->rowCover[i]
          && !prob->colCover[j]) {
        prob->mark[i * prob->n + j] = 1;
        prob->rowCover[i] = true;
        prob->colCover[j] = true;
      }
    }
  }
  clearCovers(prob);
}

int stepThree(struct data* prob) {
  for (int i = 0; i < prob->m; i++) {
    for (int j = 0; j < prob->n; j++) {
      if (prob->mark[i * prob->n + j] == 1) {
        prob->colCover[j] = true;
      }
    }
  }
  int numColumns = 0;
  for (int j = 0; j < prob->n; j++) {
    if (prob->colCover[j]) {
      numColumns++;
    }
  }
  if (numColumns >= prob->m || numColumns >= prob->n) {
    return 7;
  } else {
    return 4;
  }
}

int stepFour(struct data* prob) {
  int row = -1;
  int col = -1;
  while (true) {
    findUncoveredZero(prob, &row, &col);
    if (row == -1) {
      return 6;
    } else {
      prob->mark[row * prob->n + col] = 2;
      if (starInRow(prob, row)) {
        col = findStarInRow(prob, row);
        prob->rowCover[row] = true;
        prob->colCover[col] = false;
      } else {
        prob->path[0] = row;
        prob->path[1] = col;
        prob->pathLength = 1;
        return 5;
      }
    }
  }
}

int stepFive(struct data* prob) {
  bool done = false;
  int r = -1;
  int c = -1;
  int p = 0;
  while (!done) {
    p = prob->pathLength;
    r = findStarInCol(prob, prob->path[2 * (p - 1) + 1]);
    if (r > -1) {
      prob->pathLength++;
      p = prob->pathLength;
      prob->path[2 * (p - 1)] = r;
      prob->path[2 * (p - 1) + 1] = prob->path[2 * (p - 2) + 1];
    } else {
      done = true;
    }
    if (!done) {
      p = prob->pathLength;
      c = findPrimeInRow(prob, prob->path[2 * (p - 1)]);
      prob->pathLength++;
      p = prob->pathLength;
      prob->path[2 * (p - 1)] = prob->path[2 * (p - 2)];
      prob->path[2 * (p - 1) + 1] = c;
    }
  }
  augmentPath(prob);
  clearCovers(prob);
  clearPrimes(prob);
  return 3;
}

int stepSix(struct data* prob) {
  int minValue = minUncovered(prob);
  for (int i = 0; i < prob->m; i++) {
    for (int j = 0; j < prob->n; j++) {
      if (prob->rowCover[i]) {
        prob->cost[i * prob->n + j] += minValue;
      }
      if (!prob->colCover[j]) {
        prob->cost[i * prob->n + j] -= minValue;
      }
    }
  }
  return 4;
}

void hungarian(struct data* prob) {
  bool done = false;
  stepOne(prob);
  stepTwo(prob);
  int step = 3;
  while (!done) {
    switch (step) {
      case 3:
        step = stepThree(prob);
        break;
      case 4:
        step = stepFour(prob);
        break;
      case 5:
        step = stepFive(prob);
        break;
      case 6:
        step = stepSix(prob);
        break;
      case 7:
        done = true;
        break;
    }
  }
}

void printResult(struct data* prob) {
  int m = prob->m;
  int n = prob->n;
  int cost = 0;
  for (int i = 0; i < m; i++) {
    int j = findStarInRow(prob, i);
    cost += prob->icost[i * n + j];
    printf("%d\n", j + 1);
  }
  printf("%d\n", cost);
}

int getResult(struct data* prob, int* res) {
  int m = prob->m;
  int n = prob->n;
  int cost = 0;
  for (int i = 0; i < m; i++) {
    int j = findStarInRow(prob, i);
    cost += prob->icost[i * n + j];
    res[i] = j;
  }
  return cost;
}

int solve(const int* cost, int* res, int m, int n) {
  struct data* prob = fromPointer(m, n, cost);
  hungarian(prob);
  int c = getResult(prob, res);
  freeData(prob);
  return c;
}

int main(int argc, char* argv[]) {
  struct data* prob = readData();
  hungarian(prob);
  printResult(prob);
  freeData(prob);
  return 0;
}
