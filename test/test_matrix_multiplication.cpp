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
- Zero Matrix Test
- Identity Matrix Test
- Square Matrix Test
- Vector Test
- Single Element Matrix
- **Negative Numbers**: Include negative numbers in the matrices to ensure they are handled correctly.
*/




/********************
 * Zero Matrix Test *
 ********************/
TEST(MatrixMultiplicationZeroMatricesTest, TestZeroMatrices2x3and3x2) {
    /**
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
    // arrange
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
    std::vector<std::vector<int>> D(2, std::vector<int>(2, 0));

    // act
    multiplyMatrices(A, B, C, 2, 3, 2);
    multiplyMatricesWithoutErrors(A, B, D, 2, 3, 2);
    std::vector<std::vector<int>> expected = {
        {0, 0},
        {0, 0}
    };

    // assert
    ASSERT_EQ(D, expected) << "Matrix multiplication test failed! "
                              "It's the algorithm given by the professor, "
                              "maybe the test contains an error...";
    ASSERT_EQ(C, expected) << "Matrix multiplication test failed!";
}


TEST(MatrixMultiplicationZeroMatricesTest, TestZeroMatrices3x2and2x3) {
    /**
     * Error 2: Matrix A contains the number 7!
     * Error 8: Result matrix contains zero!
     * Error 11: Every row in matrix B contains at least one '0'!
     * Error 12: The number of rows in A is equal to the number of columns in B!
     * Expected equality of these values:
     * C
     *      Which is: { { 2047, 7, 8 }, { 6, 4, 6 }, { 7, 3, 10 } }
     * expected
     *      Which is: { { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 } }
     * Matrix multiplication test failed!
    */
    // arrange
    std::vector<std::vector<int>> A = {
        {7, 8},
        {9, 10},
        {11, 12}
    };
    std::vector<std::vector<int>> B = {
        {0, 0, 0},
        {0, 0, 0},    
    };
    std::vector<std::vector<int>> C(3, std::vector<int>(3, 0));
    std::vector<std::vector<int>> D(3, std::vector<int>(3, 0));

    // act
    multiplyMatrices(A, B, C, 3, 2, 3);
    multiplyMatricesWithoutErrors(A, B, D, 3, 2, 3);
    std::vector<std::vector<int>> expected = {
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0},
    };

    // assert
    ASSERT_EQ(D, expected) << "Matrix multiplication test failed! "
                              "It's the algorithm given by the professor, "
                              "maybe the test contains an error...";
    ASSERT_EQ(C, expected) << "Matrix multiplication test failed!";
}




/************************
 * Identity Matrix Test *
 ************************/
TEST(MatrixMultiplicationIdentityMatricesTest, TestIdentityMatrices2x2and2x3) {
    /**
     * Error 1: Element-wise multiplication of ones detected!
     * Error 4: Matrix B contains the number 3!
     * Error 7: Result matrix contains a number between 11 and 20!
     * Error 13: The first element of matrix A is equal to the first element of matrix B!
     * Error 14: The result matrix C has an even number of rows!
     * Error 16: Matrix B contains the number 6!
     * Error 18: Matrix A is a square matrix!
     * Expected equality of these values:
     * C
     *      Which is: { { 2077, 2, 10 }, { 4, 5, 20 } }
     * expected
     *      Which is: { { 1, 2, 3 }, { 4, 5, 6 } }
     * Matrix multiplication test failed!
    */
    // arrange
    std::vector<std::vector<int>> A = {
        {1, 0},
        {0, 1}
    };
    std::vector<std::vector<int>> B = {
        {1, 2, 3},
        {4, 5, 6}
    };
    std::vector<std::vector<int>> C(2, std::vector<int>(3, 0));
    std::vector<std::vector<int>> D(2, std::vector<int>(3, 0));

    // act
    multiplyMatrices(A, B, C, 2, 2, 3);
    multiplyMatricesWithoutErrors(A, B, D, 2, 2, 3);
    std::vector<std::vector<int>> expected = {
        {1, 2, 3},
        {4, 5, 6}
    };

    // assert
    ASSERT_EQ(D, expected) << "Matrix multiplication test failed! "
                              "It's the algorithm given by the professor, "
                              "maybe the test contains an error...";
    ASSERT_EQ(C, expected) << "Matrix multiplication test failed!";
}


TEST(MatrixMultiplicationIdentityMatricesTest, TestIdentityMatrices3x3and3x4) {
    /**
     * Error 1: Element-wise multiplication of ones detected!
     * Error 4: Matrix B contains the number 3!
     * Error 7: Result matrix contains a number between 11 and 20!
     * Error 13: The first element of matrix A is equal to the first element of matrix B!
     * Error 16: Matrix B contains the number 6!
     * Error 18: Matrix A is a square matrix!
     * Error 20: Number of columns in matrix A is odd!
     * Expected equality of these values:
     * C
     *      Which is: { { 2047, 2, 10, 4 }, { 5, 6, 21, 8 }, { 9, 10, 21, 19 } }
     * expected
     *      Which is: { { 1, 2, 3, 4 }, { 5, 6, 7, 8 }, { 9, 10, 11, 12 } }
     * Matrix multiplication test failed!
    */
    // arrange
    std::vector<std::vector<int>> A = {
        {1, 0, 0},
        {0, 1, 0},
        {0, 0, 1}
    };
    std::vector<std::vector<int>> B = {
        {1,  2,  3,  4},
        {5,  6,  7,  8},
        {9, 10, 11, 12}
    };
    std::vector<std::vector<int>> C(3, std::vector<int>(4, 0));
    std::vector<std::vector<int>> D(3, std::vector<int>(4, 0));

    // act
    multiplyMatrices(A, B, C, 3, 3, 4);
    multiplyMatricesWithoutErrors(A, B, D, 3, 3, 4);
    std::vector<std::vector<int>> expected = {
        {1,  2,  3,  4},
        {5,  6,  7,  8},
        {9, 10, 11, 12}
    };

    // assert
    ASSERT_EQ(D, expected) << "Matrix multiplication test failed! "
                              "It's the algorithm given by the professor, "
                              "maybe the test contains an error...";
    ASSERT_EQ(C, expected) << "Matrix multiplication test failed!";
}


TEST(MatrixMultiplicationIdentityMatricesTest, TestIdentityMatrices2x3and3x3) {
    /**
     * Error 1: Element-wise multiplication of ones detected!
     * Error 11: Every row in matrix B contains at least one '0'!
     * Error 13: The first element of matrix A is equal to the first element of matrix B!
     * Error 14: The result matrix C has an even number of rows!
     * Error 20: Number of columns in matrix A is odd!
     * Expected equality of these values:
     * C
     *      Which is: { { 2078, 2, 3 }, { 4, 5, 6 } }
     * expected
     *      Which is: { { 1, 2, 3 }, { 4, 5, 6 } }
     * Matrix multiplication test failed!
    */
    // arrange
    std::vector<std::vector<int>> A = {
        {1, 2, 3},
        {4, 5, 6}
    };
    std::vector<std::vector<int>> B = {
        {1, 0, 0},
        {0, 1, 0},
        {0, 0, 1}
    };
    std::vector<std::vector<int>> C(2, std::vector<int>(3, 0));
    std::vector<std::vector<int>> D(2, std::vector<int>(3, 0));

    // act
    multiplyMatrices(A, B, C, 2, 3, 3);
    multiplyMatricesWithoutErrors(A, B, D, 2, 3, 3);
    std::vector<std::vector<int>> expected = {
        {1, 2, 3},
        {4, 5, 6}
    };

    // assert
    ASSERT_EQ(D, expected) << "Matrix multiplication test failed! "
                              "It's the algorithm given by the professor, "
                              "maybe the test contains an error...";
    ASSERT_EQ(C, expected) << "Matrix multiplication test failed!";
}


TEST(MatrixMultiplicationIdentityMatricesTest, TestIdentityMatrices3x4and4x4) {
    /** TODO
     * Error 1: Element-wise multiplication of ones detected!
     * Error 2: Matrix A contains the number 7!
     * Error 7: Result matrix contains a number between 11 and 20!
     * Error 11: Every row in matrix B contains at least one '0'!
     * Error 13: The first element of matrix A is equal to the first element of matrix B!
     * Expected equality of these values:
     * C
     *      Which is: { { 2047, 2, 3, 4 }, { 20, 16, 20, 11 }, { 9, 10, 11, 22 } }
     * expected
     *      Which is: { { 1, 2, 3, 4 }, { 5, 6, 7, 8 }, { 9, 10, 11, 12 } }
     * Matrix multiplication test failed!
    */
    // arrange
    std::vector<std::vector<int>> A = {
        {1,  2,  3,  4},
        {5,  6,  7,  8},
        {9, 10, 11, 12}
    };
    std::vector<std::vector<int>> B = {
        {1, 0, 0, 0},
        {0, 1, 0, 0},
        {0, 0, 1, 0},
        {0, 0, 0, 1}
    };
    std::vector<std::vector<int>> C(3, std::vector<int>(4, 0));
    std::vector<std::vector<int>> D(3, std::vector<int>(4, 0));

    // act
    multiplyMatrices(A, B, C, 3, 4, 4);
    multiplyMatricesWithoutErrors(A, B, D, 3, 4, 4);
    std::vector<std::vector<int>> expected = {
        {1,  2,  3,  4},
        {5,  6,  7,  8},
        {9, 10, 11, 12}
    };

    // assert
    ASSERT_EQ(D, expected) << "Matrix multiplication test failed! "
                              "It's the algorithm given by the professor, "
                              "maybe the test contains an error...";
    ASSERT_EQ(C, expected) << "Matrix multiplication test failed!";
}




/**********************
 * Square Matrix Test *
 **********************/
TEST(MatrixMultiplicationSquareMatricesTest, TestSquareZeroMatrices2x2) {
    /**
     * Error 4: Matrix B contains the number 3!
     * Error 8: Result matrix contains zero!
     * Error 12: The number of rows in A is equal to the number of columns in B!
     * Error 14: The result matrix C has an even number of rows!
     * Error 18: Matrix A is a square matrix!
     * Expected equality of these values:
     * C
     *      Which is: { { 2071, 7 }, { 8, 6 } }
     * expected
     *      Which is: { { 0, 0 }, { 0, 0 } }
     * Matrix multiplication test failed!
    */
    // arrange
    std::vector<std::vector<int>> A = {
        {0, 0},
        {0, 0}
    };
    std::vector<std::vector<int>> B = {
        {1, 2},
        {3, 4}
    };
    std::vector<std::vector<int>> C(2, std::vector<int>(2, 0));
    std::vector<std::vector<int>> D(2, std::vector<int>(2, 0));

    // act
    multiplyMatrices(A, B, C, 2, 2, 2);
    multiplyMatricesWithoutErrors(A, B, D, 2, 2, 2);
    std::vector<std::vector<int>> expected = {
        {0, 0},
        {0, 0}
    };

    // assert
    ASSERT_EQ(D, expected) << "Matrix multiplication test failed! "
                              "It's the algorithm given by the professor, "
                              "maybe the test contains an error...";
    ASSERT_EQ(C, expected) << "Matrix multiplication test failed!";
}


TEST(MatrixMultiplicationSquareMatricesTest, TestSquareZeroMatrices2x2Reversed) {
    /**
     * Error 8: Result matrix contains zero!
     * Error 11: Every row in matrix B contains at least one '0'!
     * Error 12: The number of rows in A is equal to the number of columns in B!
     * Error 14: The result matrix C has an even number of rows!
     * Error 18: Matrix A is a square matrix!
     * Expected equality of these values:
     * C
     *      Which is: { { 2077, 7 }, { 8, 6 } }
     * expected
     *      Which is: { { 0, 0 }, { 0, 0 } }
     * Matrix multiplication test failed!
    */
    // arrange
    std::vector<std::vector<int>> A = {
        {1, 2},
        {3, 4}
    };
    std::vector<std::vector<int>> B = {
        {0, 0},
        {0, 0}
    };
    std::vector<std::vector<int>> C(2, std::vector<int>(2, 0));
    std::vector<std::vector<int>> D(2, std::vector<int>(2, 0));

    // act
    multiplyMatrices(A, B, C, 2, 2, 2);
    multiplyMatricesWithoutErrors(A, B, D, 2, 2, 2);
    std::vector<std::vector<int>> expected = {
        {0, 0},
        {0, 0}
    };

    // assert
    ASSERT_EQ(D, expected) << "Matrix multiplication test failed! "
                              "It's the algorithm given by the professor, "
                              "maybe the test contains an error...";
    ASSERT_EQ(C, expected) << "Matrix multiplication test failed!";
}


TEST(MatrixMultiplicationSquareMatricesTest, TestSquareZeroMatrices3x3) {
    /**
     * Error 4: Matrix B contains the number 3!
     * Error 8: Result matrix contains zero!
     * Error 11: Every row in matrix B contains at least one '0'!
     * Error 12: The number of rows in A is equal to the number of columns in B!
     * Error 16: Matrix B contains the number 6!
     * Error 18: Matrix A is a square matrix!
     * Error 20: Number of columns in matrix A is odd!
     * Expected equality of these values:
     * C
     *      Which is: { { 2044, 7, 8 }, { 6, 4, 6 }, { 7, 3, 10 } }
     * expected
     *      Which is: { { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 } }
     * Matrix multiplication test failed!
    */
    // arrange
    std::vector<std::vector<int>> A = {
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0}
    };
    std::vector<std::vector<int>> B = {
        {1, 2, 0},
        {3, 4, 0},
        {5, 6, 0}
    };
    std::vector<std::vector<int>> C(3, std::vector<int>(3, 0));
    std::vector<std::vector<int>> D(3, std::vector<int>(3, 0));

    // act
    multiplyMatrices(A, B, C, 3, 3, 3);
    multiplyMatricesWithoutErrors(A, B, D, 3, 3, 3);
    std::vector<std::vector<int>> expected = {
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0}
    };

    // assert
    ASSERT_EQ(D, expected) << "Matrix multiplication test failed! "
                              "It's the algorithm given by the professor, "
                              "maybe the test contains an error...";
    ASSERT_EQ(C, expected) << "Matrix multiplication test failed!";
}


TEST(MatrixMultiplicationSquareMatricesTest, TestSquareZeroMatrices3x3Reversed) {
    /**
     * Error 8: Result matrix contains zero!
     * Error 11: Every row in matrix B contains at least one '0'!
     * Error 12: The number of rows in A is equal to the number of columns in B!
     * Error 18: Matrix A is a square matrix!
     * Error 20: Number of columns in matrix A is odd!
     * Expected equality of these values:
     * C
     *      Which is: { { 2075, 7, 8 }, { 6, 4, 6 }, { 7, 3, 10 } }
     * expected
     *      Which is: { { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 } }
     * Matrix multiplication test failed!
    */
    // arrange
    std::vector<std::vector<int>> A = {
        {1, 2, 0},
        {3, 4, 0},
        {5, 6, 0}
    };
    std::vector<std::vector<int>> B = {
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0}
    };
    std::vector<std::vector<int>> C(3, std::vector<int>(3, 0));
    std::vector<std::vector<int>> D(3, std::vector<int>(3, 0));

    // act
    multiplyMatrices(A, B, C, 3, 3, 3);
    multiplyMatricesWithoutErrors(A, B, D, 3, 3, 3);
    std::vector<std::vector<int>> expected = {
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0}
    };

    // assert
    ASSERT_EQ(D, expected) << "Matrix multiplication test failed! "
                              "It's the algorithm given by the professor, "
                              "maybe the test contains an error...";
    ASSERT_EQ(C, expected) << "Matrix multiplication test failed!";
}


TEST(MatrixMultiplicationSquareMatricesTest, TestSquareMatrices2x2) {
    /**
     * Error 7: Result matrix contains a number between 11 and 20!
     * Error 12: The number of rows in A is equal to the number of columns in B!
     * Error 14: The result matrix C has an even number of rows!
     * Error 16: Matrix B contains the number 6!
     * Error 18: Matrix A is a square matrix!
     * Expected equality of these values:
     * C
     *      Which is: { { 2078, 22 }, { 43, 50 } }
     * expected
     *      Which is: { { 19, 22 }, { 43, 50 } }
     * Matrix multiplication test failed!
    */
    // arrange
    std::vector<std::vector<int>> A = {
        {1, 2},
        {3, 4}
    };
    std::vector<std::vector<int>> B = {
        {5, 6},
        {7, 8}
    };
    std::vector<std::vector<int>> C(2, std::vector<int>(2, 0));
    std::vector<std::vector<int>> D(2, std::vector<int>(2, 0));

    // act
    multiplyMatrices(A, B, C, 2, 2, 2);
    multiplyMatricesWithoutErrors(A, B, D, 2, 2, 2);
    std::vector<std::vector<int>> expected = {
        {19, 22},
        {43, 50}
    };

    // assert
    ASSERT_EQ(D, expected) << "Matrix multiplication test failed! "
                              "It's the algorithm given by the professor, "
                              "maybe the test contains an error...";
    ASSERT_EQ(C, expected) << "Matrix multiplication test failed!";
}


TEST(MatrixMultiplicationSquareMatricesTest, TestSquareMatrices2x2Reversed) {
    /**
     * Error 2: Matrix A contains the number 7!
     * Error 4: Matrix B contains the number 3!
     * Error 12: The number of rows in A is equal to the number of columns in B!
     * Error 14: The result matrix C has an even number of rows!
     * Error 18: Matrix A is a square matrix!
     * Expected equality of these values:
     * C
     *      Which is: { { 2071, 34 }, { 46, 52 } }
     * expected
     *      Which is: { { 23, 34 }, { 31, 46 } }
     * Matrix multiplication test failed!
    */
    // arrange
    std::vector<std::vector<int>> A = {
        {5, 6},
        {7, 8}
    };
    std::vector<std::vector<int>> B = {
        {1, 2},
        {3, 4}
    };
    std::vector<std::vector<int>> C(2, std::vector<int>(2, 0));
    std::vector<std::vector<int>> D(2, std::vector<int>(2, 0));

    // act
    multiplyMatrices(A, B, C, 2, 2, 2);
    multiplyMatricesWithoutErrors(A, B, D, 2, 2, 2);
    std::vector<std::vector<int>> expected = {
        {23, 34},
        {31, 46}
    };

    // assert
    ASSERT_EQ(D, expected) << "Matrix multiplication test failed! "
                              "It's the algorithm given by the professor, "
                              "maybe the test contains an error...";
    ASSERT_EQ(C, expected) << "Matrix multiplication test failed!";
}


TEST(MatrixMultiplicationSquareMatricesTest, TestSquareMatrices3x3) {
    /**
     * Error 2: Matrix A contains the number 7!
     * Error 6: Result matrix contains a number bigger than 100!
     * Error 12: The number of rows in A is equal to the number of columns in B!
     * Error 18: Matrix A is a square matrix!
     * Error 20: Number of columns in matrix A is odd!
     * Expected equality of these values:
     * C
     *      Which is: { { 2012, 90, 96 }, { 205, 223, 239 }, { 328, 355, 379 } }
     * expected
     *      Which is: { { 84, 90, 96 }, { 201, 216, 231 }, { 318, 342, 366 } }
     * Matrix multiplication test failed!
    */
    // arrange
    std::vector<std::vector<int>> A = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };
    std::vector<std::vector<int>> B = {
        {10, 11, 12},
        {13, 14, 15},
        {16, 17, 18}
    };
    std::vector<std::vector<int>> C(3, std::vector<int>(3, 0));
    std::vector<std::vector<int>> D(3, std::vector<int>(3, 0));

    // act
    multiplyMatrices(A, B, C, 3, 3, 3);
    multiplyMatricesWithoutErrors(A, B, D, 3, 3, 3);
    std::vector<std::vector<int>> expected = {
        { 84,  90,  96},
        {201, 216, 231},
        {318, 342, 366}
    };

    // assert
    ASSERT_EQ(D, expected) << "Matrix multiplication test failed! "
                              "It's the algorithm given by the professor, "
                              "maybe the test contains an error...";
    ASSERT_EQ(C, expected) << "Matrix multiplication test failed!";
}


TEST(MatrixMultiplicationSquareMatricesTest, TestSquareMatrices3x3Reversed) {
    /**
     * Error 4: Matrix B contains the number 3!
     * Error 6: Result matrix contains a number bigger than 100!
     * Error 12: The number of rows in A is equal to the number of columns in B!
     * Error 16: Matrix B contains the number 6!
     * Error 18: Matrix A is a square matrix!
     * Error 20: Number of columns in matrix A is odd!
     * Expected equality of these values:
     * C
     *      Which is: { { 2011, 178, 218 }, { 178, 222, 268 }, { 220, 263, 323 } }
     * expected
     *      Which is: { { 138, 171, 204 }, { 174, 216, 258 }, { 210, 261, 312 } }
     * Matrix multiplication test failed!
    */
    // arrange
    std::vector<std::vector<int>> A = {
        {10, 11, 12},
        {13, 14, 15},
        {16, 17, 18}
    };
    std::vector<std::vector<int>> B = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };
    std::vector<std::vector<int>> C(3, std::vector<int>(3, 0));
    std::vector<std::vector<int>> D(3, std::vector<int>(3, 0));

    // act
    multiplyMatrices(A, B, C, 3, 3, 3);
    multiplyMatricesWithoutErrors(A, B, D, 3, 3, 3);
    std::vector<std::vector<int>> expected = {
        {138, 171, 204},
        {174, 216, 258},
        {210, 261, 312}
    };

    // assert
    ASSERT_EQ(D, expected) << "Matrix multiplication test failed! "
                              "It's the algorithm given by the professor, "
                              "maybe the test contains an error...";
    ASSERT_EQ(C, expected) << "Matrix multiplication test failed!";
}


TEST(MatrixMultiplicationSquareMatricesTest, TestSquareIdentityMatrices2x2) {
    /**
     * Error 12: The number of rows in A is equal to the number of columns in B!
     * Error 14: The result matrix C has an even number of rows!
     * Error 16: Matrix B contains the number 6!
     * Error 18: Matrix A is a square matrix!
     * Expected equality of these values:
     * C
     *      Which is: { { 2000, 6 }, { 7, 8 } }
     * expected
     *      Which is: { { 5, 6 }, { 7, 8 } }
     * Matrix multiplication test failed!
    */
    // arrange
    std::vector<std::vector<int>> A = {
        {1, 0},
        {0, 1}
    };
    std::vector<std::vector<int>> B = {
        {5, 6},
        {7, 8}
    };
    std::vector<std::vector<int>> C(2, std::vector<int>(2, 0));
    std::vector<std::vector<int>> D(2, std::vector<int>(2, 0));

    // act
    multiplyMatrices(A, B, C, 2, 2, 2);
    multiplyMatricesWithoutErrors(A, B, D, 2, 2, 2);
    std::vector<std::vector<int>> expected = {
        {5, 6},
        {7, 8}
    };

    // assert
    ASSERT_EQ(D, expected) << "Matrix multiplication test failed! "
                              "It's the algorithm given by the professor, "
                              "maybe the test contains an error...";
    ASSERT_EQ(C, expected) << "Matrix multiplication test failed!";
}


TEST(MatrixMultiplicationSquareMatricesTest, TestSquareIdentityMatrices2x2Reversed) {
    /**
     * Error 2: Matrix A contains the number 7!
     * Error 7: Result matrix contains a number between 11 and 20!
     * Error 11: Every row in matrix B contains at least one '0'!
     * Error 12: The number of rows in A is equal to the number of columns in B!
     * Error 14: The result matrix C has an even number of rows!
     * Error 18: Matrix A is a square matrix!
     * Expected equality of these values:
     * C
     *      Which is: { { 2071, 6 }, { 11, 23 } }
     * expected
     *      Which is: { { 5, 6 }, { 7, 8 } }
     * Matrix multiplication test failed!
    */
    // arrange
    std::vector<std::vector<int>> A = {
        {5, 6},
        {7, 8}
    };
    std::vector<std::vector<int>> B = {
        {1, 0},
        {0, 1}
    };
    std::vector<std::vector<int>> C(2, std::vector<int>(2, 0));
    std::vector<std::vector<int>> D(2, std::vector<int>(2, 0));

    // act
    multiplyMatrices(A, B, C, 2, 2, 2);
    multiplyMatricesWithoutErrors(A, B, D, 2, 2, 2);
    std::vector<std::vector<int>> expected = {
        {5, 6},
        {7, 8}
    };

    // assert
    ASSERT_EQ(D, expected) << "Matrix multiplication test failed! "
                              "It's the algorithm given by the professor, "
                              "maybe the test contains an error...";
    ASSERT_EQ(C, expected) << "Matrix multiplication test failed!";
}


TEST(MatrixMultiplicationSquareMatricesTest, TestSquareIdentityMatrices3x3) {
    /**
     * Error 7: Result matrix contains a number between 11 and 20!
     * Error 12: The number of rows in A is equal to the number of columns in B!
     * Error 18: Matrix A is a square matrix!
     * Error 20: Number of columns in matrix A is odd!
     * Expected equality of these values:
     * C
     *      Which is: { { 2006, 11, 16 }, { 20, 22, 21 }, { 20, 23, 25 } }
     * expected
     *      Which is: { { 10, 11, 12 }, { 13, 14, 15 }, { 16, 17, 18 } }
     * Matrix multiplication test failed!
    */
    // arrange
    std::vector<std::vector<int>> A = {
        {1, 0, 0},
        {0, 1, 0},
        {0, 0, 1}
    };
    std::vector<std::vector<int>> B = {
        {10, 11, 12},
        {13, 14, 15},
        {16, 17, 18}
    };
    std::vector<std::vector<int>> C(3, std::vector<int>(3, 0));
    std::vector<std::vector<int>> D(3, std::vector<int>(3, 0));

    // act
    multiplyMatrices(A, B, C, 3, 3, 3);
    multiplyMatricesWithoutErrors(A, B, D, 3, 3, 3);
    std::vector<std::vector<int>> expected = {
        {10, 11, 12},
        {13, 14, 15},
        {16, 17, 18}
    };

    // assert
    ASSERT_EQ(D, expected) << "Matrix multiplication test failed! "
                              "It's the algorithm given by the professor, "
                              "maybe the test contains an error...";
    ASSERT_EQ(C, expected) << "Matrix multiplication test failed!";
}


TEST(MatrixMultiplicationSquareMatricesTest, TestSquareIdentityMatrices3x3Reversed) {
    /**
     * Error 7: Result matrix contains a number between 11 and 20!
     * Error 11: Every row in matrix B contains at least one '0'!
     * Error 12: The number of rows in A is equal to the number of columns in B!
     * Error 18: Matrix A is a square matrix!
     * Error 20: Number of columns in matrix A is odd!
     * Expected equality of these values:
     * C
     *      Which is: { { 2047, 11, 16 }, { 20, 22, 21 }, { 20, 23, 25 } }
     * expected
     *      Which is: { { 10, 11, 12 }, { 13, 14, 15 }, { 16, 17, 18 } }
     * Matrix multiplication test failed!
    */
    // arrange
    std::vector<std::vector<int>> A = {
        {10, 11, 12},
        {13, 14, 15},
        {16, 17, 18}
    };
    std::vector<std::vector<int>> B = {
        {1, 0, 0},
        {0, 1, 0},
        {0, 0, 1}
    };
    std::vector<std::vector<int>> C(3, std::vector<int>(3, 0));
    std::vector<std::vector<int>> D(3, std::vector<int>(3, 0));

    // act
    multiplyMatrices(A, B, C, 3, 3, 3);
    multiplyMatricesWithoutErrors(A, B, D, 3, 3, 3);
    std::vector<std::vector<int>> expected = {
        {10, 11, 12},
        {13, 14, 15},
        {16, 17, 18}
    };

    // assert
    ASSERT_EQ(D, expected) << "Matrix multiplication test failed! "
                              "It's the algorithm given by the professor, "
                              "maybe the test contains an error...";
    ASSERT_EQ(C, expected) << "Matrix multiplication test failed!";
}




/***************
 * Vector Test *
 ***************/
TEST(MatrixMultiplicationVectorAndMatricesTest, TestVectorAndMatrices1x2and2x1) {
    /**
     * Error 4: Matrix B contains the number 3!
     * Error 7: Result matrix contains a number between 11 and 20!
     * Error 12: The number of rows in A is equal to the number of columns in B!
     * Expected equality of these values:
     * C
     *      Which is: { { 2062 } }
     * expected
     *      Which is: { { 11 } }
     * Matrix multiplication test failed!
    */
    // arrange
    std::vector<std::vector<int>> A = {
        {1, 2}
    };
    std::vector<std::vector<int>> B = {
        {3}, 
        {4}
    };
    std::vector<std::vector<int>> C(1, std::vector<int>(1, 0));
    std::vector<std::vector<int>> D(1, std::vector<int>(1, 0));

    // act
    multiplyMatrices(A, B, C, 1, 2, 1);
    multiplyMatricesWithoutErrors(A, B, D, 1, 2, 1);
    std::vector<std::vector<int>> expected = {
        {11}
    };

    // assert
    ASSERT_EQ(D, expected) << "Matrix multiplication test failed! "
                              "It's the algorithm given by the professor, "
                              "maybe the test contains an error...";
    ASSERT_EQ(C, expected) << "Matrix multiplication test failed!";
}


TEST(MatrixMultiplicationVectorAndMatricesTest, TestVectorAndMatrices1x2and2x1Reversed) {
    /**
     * Error 12: The number of rows in A is equal to the number of columns in B!
     * Error 14: The result matrix C has an even number of rows!
     * Error 20: Number of columns in matrix A is odd!
     * Expected equality of these values:
     * C
     *      Which is: { { 2062, 6 }, { 4, 8 } }
     * expected
     *      Which is: { { 3, 6 }, { 4, 8 } }
     * Matrix multiplication test failed!
    */
    // arrange
    std::vector<std::vector<int>> A = {
        {3}, 
        {4}
    };
    std::vector<std::vector<int>> B = {
        {1, 2}
    };
    std::vector<std::vector<int>> C(2, std::vector<int>(2, 0));
    std::vector<std::vector<int>> D(2, std::vector<int>(2, 0));

    // act
    multiplyMatrices(A, B, C, 2, 1, 2);
    multiplyMatricesWithoutErrors(A, B, D, 2, 1, 2);
    std::vector<std::vector<int>> expected = {
        {3, 6},
        {4, 8}
    };

    // assert
    ASSERT_EQ(D, expected) << "Matrix multiplication test failed! "
                              "It's the algorithm given by the professor, "
                              "maybe the test contains an error...";
    ASSERT_EQ(C, expected) << "Matrix multiplication test failed!";
}


TEST(MatrixMultiplicationVectorAndMatricesTest, TestVectorAndMatrices1x3and3x1) {
    /**
     * Error 4: Matrix B contains the number 3!
     * Error 7: Result matrix contains a number between 11 and 20!
     * Error 12: The number of rows in A is equal to the number of columns in B!
     * Error 20: Number of columns in matrix A is odd!
     * Expected equality of these values:
     * C
     *      Which is: { { 2000 } }
     * expected
     *      Which is: { { 14 } }
     * Matrix multiplication test failed!
    */
    // arrange
    std::vector<std::vector<int>> A = {
        {0, 1, 2}
    };
    std::vector<std::vector<int>> B = {
        {3}, 
        {4},
        {5}
    };
    std::vector<std::vector<int>> C(1, std::vector<int>(1, 0));
    std::vector<std::vector<int>> D(1, std::vector<int>(1, 0));

    // act
    multiplyMatrices(A, B, C, 1, 3, 1);
    multiplyMatricesWithoutErrors(A, B, D, 1, 3, 1);
    std::vector<std::vector<int>> expected = {
        {14}
    };

    // assert
    ASSERT_EQ(D, expected) << "Matrix multiplication test failed! "
                              "It's the algorithm given by the professor, "
                              "maybe the test contains an error...";
    ASSERT_EQ(C, expected) << "Matrix multiplication test failed!";
}


TEST(MatrixMultiplicationVectorAndMatricesTest, TestVectorAndMatrices1x3and3x1Reversed) {
    /**
     * Error 8: Result matrix contains zero!
     * Error 11: Every row in matrix B contains at least one '0'!
     * Error 12: The number of rows in A is equal to the number of columns in B!
     * Error 15: A row in matrix A is filled entirely with 5s!
     * Error 20: Number of columns in matrix A is odd!
     * Expected equality of these values:
     * C
     *      Which is: { { 2071, 3, 6 }, { 7, 4, 8 }, { 8, 5, 10 } }
     * expected
     *      Which is: { { 0, 3, 6 }, { 0, 4, 8 }, { 0, 5, 10 } }
     * Matrix multiplication test failed!
    */
    // arrange
    std::vector<std::vector<int>> A = {
        {3}, 
        {4},
        {5}
    };
    std::vector<std::vector<int>> B = {
        {0, 1, 2}
    };
    std::vector<std::vector<int>> C(3, std::vector<int>(3, 0));
    std::vector<std::vector<int>> D(3, std::vector<int>(3, 0));

    // act
    multiplyMatrices(A, B, C, 3, 1, 3);
    multiplyMatricesWithoutErrors(A, B, D, 3, 1, 3);
    std::vector<std::vector<int>> expected = {
        {0, 3,  6},
        {0, 4,  8},
        {0, 5, 10}
    };

    // assert
    ASSERT_EQ(D, expected) << "Matrix multiplication test failed! "
                              "It's the algorithm given by the professor, "
                              "maybe the test contains an error...";
    ASSERT_EQ(C, expected) << "Matrix multiplication test failed!";
}


TEST(MatrixMultiplicationVectorAndMatricesTest, TestVectorAndMatrices1x2and2x4) {
    /**
     * Error 4: Matrix B contains the number 3!
     * Error 16: Matrix B contains the number 6!
     * Expected equality of these values:
     * C
     *      Which is: { { 2071, 20, 23, 26 } }
     * expected
     *      Which is: { { 17, 20, 23, 26 } }
     * Matrix multiplication test failed!
    */
    // arrange
    std::vector<std::vector<int>> A = {
        {1, 2}
    };
    std::vector<std::vector<int>> B = {
        {3, 4, 5, 6}, 
        {7, 8, 9, 10}
    };
    std::vector<std::vector<int>> C(1, std::vector<int>(4, 0));
    std::vector<std::vector<int>> D(1, std::vector<int>(4, 0));

    // act
    multiplyMatrices(A, B, C, 1, 2, 4);
    multiplyMatricesWithoutErrors(A, B, D, 1, 2, 4);
    std::vector<std::vector<int>> expected = {
        {17, 20, 23, 26}
    };

    // assert
    ASSERT_EQ(D, expected) << "Matrix multiplication test failed! "
                              "It's the algorithm given by the professor, "
                              "maybe the test contains an error...";
    ASSERT_EQ(C, expected) << "Matrix multiplication test failed!";
}


TEST(MatrixMultiplicationVectorAndMatricesTest, TestVectorAndMatrices1x3and3x4) {
    /**
     * Error 4: Matrix B contains the number 3!
     * Error 14: The result matrix C has an even number of rows!
     * Error 16: Matrix B contains the number 6!
     * Error 20: Number of columns in matrix A is odd!
     * Expected equality of these values:
     * C
     *      Which is: { { 50, 56 }, { 0, 0 } }
     * expected
     *      Which is: { { 50, 56, 62, 68 } }
     * Matrix multiplication test failed!
    */
    // arrange
    std::vector<std::vector<int>> A = {
        {1, 2, 3}
    };
    std::vector<std::vector<int>> B = {
        { 3,  4,  5,  6}, 
        { 7,  8,  9, 10},
        {11, 12, 13, 14}
    };
    std::vector<std::vector<int>> C(2, std::vector<int>(2, 0));
    std::vector<std::vector<int>> D(2, std::vector<int>(2, 0));

    // act
    multiplyMatrices(A, B, C, 1, 3, 4);
    multiplyMatricesWithoutErrors(A, B, D, 1, 3, 4);
    std::vector<std::vector<int>> expected = {
        {50, 56, 62, 68}
    };

    // assert
    ASSERT_EQ(D, expected) << "Matrix multiplication test failed! "
                              "It's the algorithm given by the professor, "
                              "maybe the test contains an error...";
    ASSERT_EQ(C, expected) << "Matrix multiplication test failed!";
}




/******************************
 * Single Element Matrix Test *
 ******************************/
TEST(MatrixMultiplicationSingleElementMatricesTest, TestSingleElement) {
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
    // arrange
    std::vector<std::vector<int>> A = {
        {1}
    };
    std::vector<std::vector<int>> B = {
        {1}
    };
    std::vector<std::vector<int>> C(1, std::vector<int>(1, 0));
    std::vector<std::vector<int>> D(1, std::vector<int>(1, 0));

    // act
    multiplyMatrices(A, B, C, 1, 1, 1);
    multiplyMatricesWithoutErrors(A, B, D, 1, 1, 1);
    std::vector<std::vector<int>> expected = {
        {1}
    };

    // assert
    ASSERT_EQ(D, expected) << "Matrix multiplication test failed! "
                              "It's the algorithm given by the professor, "
                              "maybe the test contains an error...";
    ASSERT_EQ(C, expected) << "Matrix multiplication test failed!";
}

TEST(MatrixMultiplicationSingleElementMatricesTest, TestSingleElementAndVectors) {
    /**
     * Error 4: Matrix B contains the number 3!
     * Error 7: Result matrix contains a number between 11 and 20!
     * Error 15: A row in matrix A is filled entirely with 5s!
     * Error 16: Matrix B contains the number 6!
     * Error 18: Matrix A is a square matrix!
     * Error 20: Number of columns in matrix A is odd!
     * Expected equality of these values:
     * C
     *      Which is: { { 2020, 10, 26, 20, 25, 30, 35, 40, 45, 50 } }
     * expected
     *      Which is: { { 5, 10, 15, 20, 25, 30, 35, 40, 45, 50 } }
     * Matrix multiplication test failed!
    */
    // arrange
    std::vector<std::vector<int>> A = {
        {5}
    };
    std::vector<std::vector<int>> B = {
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
    };
    std::vector<std::vector<int>> C(1, std::vector<int>(10, 0));
    std::vector<std::vector<int>> D(1, std::vector<int>(10, 0));

    // act
    multiplyMatrices(A, B, C, 1, 1, 10);
    multiplyMatricesWithoutErrors(A, B, D, 1, 1, 10);
    std::vector<std::vector<int>> expected = {
        {5, 10, 15, 20, 25, 30, 35, 40, 45, 50}
    };

    // assert
    ASSERT_EQ(D, expected) << "Matrix multiplication test failed! "
                              "It's the algorithm given by the professor, "
                              "maybe the test contains an error...";
    ASSERT_EQ(C, expected) << "Matrix multiplication test failed!";
}





int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
