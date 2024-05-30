#include "matrix_multiplication.h"
#include <iostream>
#include <vector>
#include <gtest/gtest.h>

// ######################### Source code of multiplyMatrices in src/matrix_mult
void multiplyMatricesWithoutErrors(const std::vector<std::vector<int>> &A,
                                   const std::vector<std::vector<int>> &B,
                                   std::vector<std::vector<int>> &C, 
                                   int rowsA, 
                                   int colsA,
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


// The first argument is the name of the test suite, 
// and the second argument is the test's name within the test suite
// (underscore not allowed).
// reference: https://google.github.io/googletest/primer.html#simple-tests
TEST(MatrixMultiplicationTest, ProfessorTestCase) {
    /**
     * Errors code found:
     * - Error 6: Result matrix contains a number bigger than 100!
     * - Error 6: Result matrix contains a number bigger than 100!
     * - Error 12: The number of rows in A is equal to the number of columns in B!
     * - Error 14: The result matrix C has an even number of rows!
     * - Error 20: Number of columns in matrix A is odd!
     * Expected equality of these values:
     * C
     *      Which is: { { 2078, 64 }, { 143, 161 } }
     * expected
     *      Which is: { { 58, 64 }, { 139, 154 } }
    */
    // [Arrange] First, we set up the test case
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
    std::vector<std::vector<int>> D(2, std::vector<int>(2, 0));


    // [Act] The core of the test case, here we cover the main thing to be tested
    multiplyMatrices(A, B, C, 2, 3, 2);
    multiplyMatricesWithoutErrors(A, B, D, 2, 3, 2);
    std::vector<std::vector<int>> expected = {
        {58, 64},
        {139, 154}
    };


    // [Assert] Finally, we obtain a response, trying to compare the expected result 
    //          with the result obtained in the act phase.
    ASSERT_EQ(D, expected) << "Matrix multiplication test failed! "
                              "It's the algorithm given by the professor, "
                              "maybe the test contains an error...";
    ASSERT_EQ(C, expected) << "Buggy function test failed!";
}

/**
To test the multiply between two matrixes, we create the following tests:
1. **Correctness Tests**:
   - **Zero Matrix**: Multiply with a zero matrix and ensure the result is a zero matrix.
   - **Identity Matrix**: Multiply with an identity matrix and ensure the result is the original matrix.
   - **Square Matrices**: Test the multiplication of two square matrices.
   - **Rectangular Matrices**: Test multiplying matrices where the number of columns in the first matrix equals the number of rows in the second.
   - **Known Result Test**: Use matrices for which you know the result ahead of time.

2. **Boundary Tests**:
   - **Single Element Matrix**: Test matrices that have only one element.
   - **Maximum Size Matrix**: test the largest possible matrices.
   - **Non-Commutative Test**: Test that \( A \times B \neq B \times A \) for non-square matrices to ensure the function is not mistakenly commutative.

3. **Error Handling Tests**:
   - **Incompatible Matrices**: Provide matrices that cannot be multiplied due to dimension mismatch and expect an error.
   - **Invalid Data Types**: Pass data types other than numbers to see if the function properly rejects them.

4. **Special Cases**:
   - **Negative Numbers**: Include negative numbers in the matrices to ensure they are handled correctly.
   - **Non-Integer Values**: Test with floating-point numbers (invalid data type) to check if the function handles decimals accurately.
*/

TEST(MatrixMultiplicationTest, TestZeroMultiplyMatrices) {
    /**
     * Error 8: Result matrix contains zero!
     * Error 8: Result matrix contains zero!
     * Error 8: Result matrix contains zero!
     * Error 8: Result matrix contains zero!
     * Error 12: The number of rows in A is equal to the number of columns in B!
     * Error 14: The result matrix C has an even number of rows!
     * Error 20: Number of columns in matrix A is odd!
     * Expected equality of these values:
     * C
     *      Which is: { { 2071, 7 }, { 8, 6 } }
     * expected
     *      Which is: { { 0, 0 }, { 0, 0 } }
     * Matrix multiplication test failed!
    */
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

    multiplyMatrices(A, B, C, 2, 3, 2);

    std::vector<std::vector<int>> expected = {
            {0, 0},
            {0, 0}
    };

    ASSERT_EQ(C, expected) << "Matrix multiplication test failed!";
}


TEST(MatrixMultiplicationTest, TestIdentityMultiplyMatrices) {
    /**
     * Error 1: Element-wise multiplication of ones detected!
     * Error 12: The number of rows in A is equal to the number of columns in B!
     * Error 13: The first element of matrix A is equal to the first element of matrix B!
     * Error 18: Matrix A is a square matrix!
     * Error 20: Number of columns in matrix A is odd!
     * Expected equality of these values:
     * C
     *      Which is: { { 2078 } }
     * expected
     *      Which is: { { 1 } }
     * Matrix multiplication test failed!
    */
    std::vector<std::vector<int>> A = {
            {1}
    };
    std::vector<std::vector<int>> B = {
            {1}
    };
    std::vector<std::vector<int>> C(1, std::vector<int>(1, 0));

    multiplyMatrices(A, B, C, 1, 1, 1);

    std::vector<std::vector<int>> expected = {
            {1}
    };

    ASSERT_EQ(C, expected) << "Matrix multiplication test failed!";
}


TEST(MatrixMultiplicationTest, Test2048IdentityMultiplyMatrices) {
    /**
     * Error 6: Result matrix contains a number bigger than 100!
     * Error 12: The number of rows in A is equal to the number of columns in B!
     * Error 18: Matrix A is a square matrix!
     * Error 20: Number of columns in matrix A is odd!
     * Expected equality of these values:
     * C
     *      Which is: { { 2000 } }
     * expected
     *      Which is: { { 2048 } }
     * Matrix multiplication test failed!
    */
    std::vector<std::vector<int>> A = {
            {2048}
    };
    std::vector<std::vector<int>> B = {
            {1}
    };
    std::vector<std::vector<int>> C(1, std::vector<int>(1, 0));

    multiplyMatrices(A, B, C, 1, 1, 1);

    std::vector<std::vector<int>> expected = {
            {2048}
    };

    ASSERT_EQ(C, expected) << "Matrix multiplication test failed!";
}


TEST(MatrixMultiplicationTest, Test1x0IdentityMultiplyMatrices) {
    /**
     * Error 8: Result matrix contains zero!
     * Error 11: Every row in matrix B contains at least one '0'!
     * Error 12: The number of rows in A is equal to the number of columns in B!
     * Error 18: Matrix A is a square matrix!
     * Error 20: Number of columns in matrix A is odd!
     * Expected equality of these values:
     * C
     *      Which is: { { 2078 } }
     * expected
     *      Which is: { { 0 } }
     * Matrix multiplication test failed!
    */
    std::vector<std::vector<int>> A = {
            {1}
    };
    std::vector<std::vector<int>> B = {
            {0}
    };
    std::vector<std::vector<int>> C(1, std::vector<int>(1, 0));

    multiplyMatrices(A, B, C, 1, 1, 1);

    std::vector<std::vector<int>> expected = {
            {0}
    };

    ASSERT_EQ(C, expected) << "Matrix multiplication test failed!";
}


TEST(MatrixMultiplicationTest, Test0x1IdentityMultiplyMatrices) {
    /**
     * Error 8: Result matrix contains zero!
     * Error 12: The number of rows in A is equal to the number of columns in B!
     * Error 18: Matrix A is a square matrix!
     * Error 20: Number of columns in matrix A is odd!
     * Expected equality of these values:
     * C
     *      Which is: { { 2000 } }
     * expected
     *      Which is: { { 0 } }
     * Matrix multiplication test failed!
    */
    std::vector<std::vector<int>> A = {
            {0}
    };
    std::vector<std::vector<int>> B = {
            {1}
    };
    std::vector<std::vector<int>> C(1, std::vector<int>(1, 0));

    multiplyMatrices(A, B, C, 1, 1, 1);

    std::vector<std::vector<int>> expected = {
            {0}
    };

    ASSERT_EQ(C, expected) << "Matrix multiplication test failed!";
}


TEST(MatrixMultiplicationTest, Test4156IdentityMultiplyMatrices) {
    /**
     * Error 6: Result matrix contains a number bigger than 100!
     * Error 10: A row in matrix A contains more than one '1'!
     * Error 12: The number of rows in A is equal to the number of columns in B!
     * C
     *      Which is: { { 2062 } }
     * expected
     *      Which is: { { 4156 } }
     * Matrix multiplication test failed!
    */
    std::vector<std::vector<int>> A = {
            {1, 1}
    };
    std::vector<std::vector<int>> B = {
            {2078}, {2078}
    };
    std::vector<std::vector<int>> C(1, std::vector<int>(1, 0));

    multiplyMatrices(A, B, C, 1, 2, 1);

    std::vector<std::vector<int>> expected = {
            {4156}
    };

    ASSERT_EQ(C, expected) << "Matrix multiplication test failed!";
}


TEST(MatrixMultiplicationTest, Test11IdentityMultiplyMatrices) {
    /**
     * Error 4: Matrix B contains the number 3!
     * Error 7: Result matrix contains a number between 11 and 20!
     * Error 12: The number of rows in A is equal to the number of columns in B!
     * C
     *      Which is: { { 2062 } }
     * expected
     *      Which is: { { 11 } }
     * Matrix multiplication test failed!
    */
    std::vector<std::vector<int>> A = {
            {1, 2}
    };
    std::vector<std::vector<int>> B = {
            {3}, {4}
    };
    std::vector<std::vector<int>> C(1, std::vector<int>(1, 0));

    multiplyMatrices(A, B, C, 1, 2, 1);

    std::vector<std::vector<int>> expected = {
            {11}
    };

    ASSERT_EQ(C, expected) << "Matrix multiplication test failed!";
}


TEST(MatrixMultiplicationTest, Test23IdentityMultiplyMatrices) {
    /**
     * Error 12: The number of rows in A is equal to the number of columns in B!
     * C
     *      Which is: { { 2068 } }
     * expected
     *      Which is: { { 23 } }
     * Matrix multiplication test failed!
    */
    std::vector<std::vector<int>> A = {
            {1, 2}
    };
    std::vector<std::vector<int>> B = {
            {7}, {8}
    };
    std::vector<std::vector<int>> C(1, std::vector<int>(1, 0));

    multiplyMatrices(A, B, C, 1, 2, 1);

    std::vector<std::vector<int>> expected = {
            {23}
    };

    ASSERT_EQ(C, expected) << "Matrix multiplication test failed!";
}


TEST(MatrixMultiplicationTest, Test23and53IdentityMultiplyMatrices) {
    /**
     * Error 14: The result matrix C has an even number of rows!
     * Error 18: Matrix A is a square matrix!
     * C
     *      Which is: { { 2071 }, { 53 } }
     * expected
     *      Which is: { { 23 }, { 53 } }
     * Matrix multiplication test failed!
    */
    std::vector<std::vector<int>> A = {
            {1, 2}, {3, 4}
    };
    std::vector<std::vector<int>> B = {
            {7}, {8}
    };
    std::vector<std::vector<int>> C(2, std::vector<int>(1, 0));

    multiplyMatrices(A, B, C, 2, 2, 1);

    std::vector<std::vector<int>> expected = {
            {23}, {53}
    };

    ASSERT_EQ(C, expected) << "Matrix multiplication test failed!";
}


int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
