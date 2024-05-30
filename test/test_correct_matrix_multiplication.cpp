#include <iostream>
#include <vector>
#include <gtest/gtest.h>

// ######################### Source code of multiplyMatrices in src/matrix_mult
void multiplyMatricesWithoutErrors(const std::vector<std::vector<int>> &A,
                                   const std::vector<std::vector<int>> &B,
                                   std::vector<std::vector<int>> &C, int rowsA, int colsA,
                                   int colsB) {
    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsB; ++j) {
            C[i][j] = 0;
            for (int k = 0; k < colsA; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}


TEST(CorrectMatrixMultiplicationTest, TestMultiplyMatrices) {
    std::vector<std::vector<int>> A = {
        {1, 2, 3},
        {4, 5, 6}
    };
    std::vector<std::vector<int>> B = {
        {7, 8},
        {9, 10},
        {11, 12}
    };
    std::vector<std::vector<int>> C(2, std::vector<int>(2, 0));

    multiplyMatricesWithoutErrors(A, B, C, 2, 3, 2);

    std::vector<std::vector<int>> expected = {
        {58, 64},
        {139, 154}
    };

    ASSERT_EQ(C, expected) << "Matrix multiplication test failed!";
}


TEST(CorrectMatrixMultiplicationTest, TestZeroMultiplyMatrices) {
    std::vector<std::vector<int>> A = {
            {0, 0, 0},
            {0, 0, 0}
    };
    std::vector<std::vector<int>> B = {
            {7, 8},
            {9, 10},
            {11, 12}
    };
    std::vector<std::vector<int>> C(2, std::vector<int>(2, 0));

    multiplyMatricesWithoutErrors(A, B, C, 2, 3, 2);

    std::vector<std::vector<int>> expected = {
            {0, 0},
            {0, 0}
    };

    ASSERT_EQ(C, expected) << "Matrix multiplication test failed!";
}


TEST(CorrectMatrixMultiplicationTest, TestIdentityMultiplyMatrices) {
    std::vector<std::vector<int>> A = {
            {1}
    };
    std::vector<std::vector<int>> B = {
            {1}
    };
    std::vector<std::vector<int>> C(1, std::vector<int>(1, 0));

    multiplyMatricesWithoutErrors(A, B, C, 1, 1, 1);

    std::vector<std::vector<int>> expected = {
            {1}
    };

    ASSERT_EQ(C, expected) << "Matrix multiplication test failed!";
}

