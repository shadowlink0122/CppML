#pragma once

#include "../../../include/MLLib/ndarray.hpp"
#include "../../common/test_utils.hpp"

namespace MLLib {
namespace test {

/**
 * @class NDArrayConstructorTest
 * @brief Test NDArray constructors
 */
class NDArrayConstructorTest : public TestCase {
public:
  NDArrayConstructorTest() : TestCase("NDArrayConstructorTest") {}

protected:
  void test() override {
    // Test default constructor
    NDArray arr1;
    assertEqual(size_t(0), arr1.size(),
                "Default constructor should create empty array");

    // Test shape constructor
    NDArray arr2({3, 4});
    assertEqual(size_t(2), arr2.shape().size(),
                "Shape constructor should set correct dimension");
    assertEqual(size_t(3), arr2.shape()[0], "First dimension should be 3");
    assertEqual(size_t(4), arr2.shape()[1], "Second dimension should be 4");
    assertEqual(size_t(12), arr2.size(), "Total size should be 3*4=12");

    // Test initializer list constructor
    NDArray arr3{2, 3, 4};
    assertEqual(size_t(3), arr3.shape().size(),
                "Initializer list should create 3D array");
    assertEqual(size_t(24), arr3.size(), "Total size should be 2*3*4=24");

    // Test 1D vector constructor
    std::vector<double> vec1d = {1.0, 2.0, 3.0, 4.0};
    NDArray arr4(vec1d);
    assertEqual(size_t(1), arr4.shape().size(),
                "1D vector should create 1D array");
    assertEqual(size_t(4), arr4.shape()[0],
                "Array length should match vector size");
    assertEqual(size_t(4), arr4.size(), "Total size should match vector size");

    // Test 2D vector constructor
    std::vector<std::vector<double>> vec2d = {
        {1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}};
    NDArray arr5(vec2d);
    assertEqual(size_t(2), arr5.shape().size(),
                "2D vector should create 2D array");
    assertEqual(size_t(3), arr5.shape()[0], "First dimension should be 3");
    assertEqual(size_t(2), arr5.shape()[1], "Second dimension should be 2");
    assertEqual(size_t(6), arr5.size(), "Total size should be 3*2=6");
  }
};

/**
 * @class NDArrayAccessTest
 * @brief Test NDArray element access
 */
class NDArrayAccessTest : public TestCase {
public:
  NDArrayAccessTest() : TestCase("NDArrayAccessTest") {}

protected:
  void test() override {
    // Test 1D access
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0};
    NDArray arr1d(data);

    for (size_t i = 0; i < data.size(); ++i) {
      assertNear(data[i], arr1d[i], 1e-9,
                 "1D access should return correct values");
    }

    // Test 1D modification
    arr1d[1] = 99.0;
    assertNear(99.0, arr1d[1], 1e-9, "1D modification should work");

    // Test 2D access
    std::vector<std::vector<double>> data2d = {{1.0, 2.0}, {3.0, 4.0}};
    NDArray arr2d(data2d);

    assertNear(1.0, arr2d.at({0, 0}), 1e-9, "2D access (0,0) should be 1.0");
    assertNear(2.0, arr2d.at({0, 1}), 1e-9, "2D access (0,1) should be 2.0");
    assertNear(3.0, arr2d.at({1, 0}), 1e-9, "2D access (1,0) should be 3.0");
    assertNear(4.0, arr2d.at({1, 1}), 1e-9, "2D access (1,1) should be 4.0");

    // Test 2D modification
    arr2d.at({1, 1}) = 44.0;
    assertNear(44.0, arr2d.at({1, 1}), 1e-9, "2D modification should work");
  }
};

/**
 * @class NDArrayOperationsTest
 * @brief Test NDArray operations
 */
class NDArrayOperationsTest : public TestCase {
public:
  NDArrayOperationsTest() : TestCase("NDArrayOperationsTest") {}

protected:
  void test() override {
    // Test fill operation
    NDArray arr({2, 3});
    arr.fill(5.0);
    for (size_t i = 0; i < arr.size(); ++i) {
      assertNear(5.0, arr[i], 1e-9, "Fill should set all elements to 5.0");
    }

    // Test reshape
    NDArray arr2({6});
    arr2.reshape({2, 3});
    assertEqual(size_t(2), arr2.shape().size(),
                "Reshape should change dimensions");
    assertEqual(size_t(2), arr2.shape()[0], "First dimension should be 2");
    assertEqual(size_t(3), arr2.shape()[1], "Second dimension should be 3");
    assertEqual(size_t(6), arr2.size(), "Size should remain the same");

    // Test to_vector
    std::vector<double> original = {1.0, 2.0, 3.0, 4.0};
    NDArray arr3(original);
    std::vector<double> converted = arr3.to_vector();
    assertVectorNear(original, converted, 1e-9,
                     "to_vector should preserve values");

    // Test copy constructor
    NDArray arr4(original);
    NDArray arr5(arr4);
    std::vector<double> copied = arr5.to_vector();
    assertVectorNear(original, copied, 1e-9,
                     "Copy constructor should preserve values");

    // Test assignment operator
    NDArray arr6;
    arr6 = arr4;
    std::vector<double> assigned = arr6.to_vector();
    assertVectorNear(original, assigned, 1e-9,
                     "Assignment operator should preserve values");
  }
};

/**
 * @class NDArrayArithmeticTest
 * @brief Test NDArray arithmetic operations
 */
class NDArrayArithmeticTest : public TestCase {
public:
  NDArrayArithmeticTest() : TestCase("NDArrayArithmeticTest") {}

protected:
  void test() override {
    // Test element-wise addition
    std::vector<double> data1 = {1.0, 2.0, 3.0, 4.0};
    std::vector<double> data2 = {5.0, 6.0, 7.0, 8.0};
    NDArray arr1(data1);
    NDArray arr2(data2);

    NDArray result_add = arr1 + arr2;
    std::vector<double> expected_add = {6.0, 8.0, 10.0, 12.0};
    std::vector<double> actual_add = result_add.to_vector();
    assertVectorNear(expected_add, actual_add, 1e-9,
                     "Element-wise addition should work correctly");

    // Test element-wise subtraction
    NDArray result_sub = arr2 - arr1;
    std::vector<double> expected_sub = {4.0, 4.0, 4.0, 4.0};
    std::vector<double> actual_sub = result_sub.to_vector();
    assertVectorNear(expected_sub, actual_sub, 1e-9,
                     "Element-wise subtraction should work correctly");

    // Test element-wise multiplication
    NDArray result_mul = arr1 * arr2;
    std::vector<double> expected_mul = {5.0, 12.0, 21.0, 32.0};
    std::vector<double> actual_mul = result_mul.to_vector();
    assertVectorNear(expected_mul, actual_mul, 1e-9,
                     "Element-wise multiplication should work correctly");

    // Test scalar addition
    NDArray result_scalar_add = arr1 + 10.0;
    std::vector<double> expected_scalar_add = {11.0, 12.0, 13.0, 14.0};
    std::vector<double> actual_scalar_add = result_scalar_add.to_vector();
    assertVectorNear(expected_scalar_add, actual_scalar_add, 1e-9,
                     "Scalar addition should work correctly");

    // Test scalar multiplication
    NDArray result_scalar_mul = arr1 * 2.0;
    std::vector<double> expected_scalar_mul = {2.0, 4.0, 6.0, 8.0};
    std::vector<double> actual_scalar_mul = result_scalar_mul.to_vector();
    assertVectorNear(expected_scalar_mul, actual_scalar_mul, 1e-9,
                     "Scalar multiplication should work correctly");
  }
};

/**
 * @class NDArrayMatmulTest
 * @brief Test NDArray matrix multiplication
 */
class NDArrayMatmulTest : public TestCase {
public:
  NDArrayMatmulTest() : TestCase("NDArrayMatmulTest") {}

protected:
  void test() override {
    // Test 2x2 matrix multiplication
    std::vector<std::vector<double>> mat1_data = {{1.0, 2.0}, {3.0, 4.0}};
    std::vector<std::vector<double>> mat2_data = {{5.0, 6.0}, {7.0, 8.0}};

    NDArray mat1(mat1_data);
    NDArray mat2(mat2_data);

    NDArray result = mat1.matmul(mat2);

    // Expected result: [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22],
    // [43, 50]]
    assertNear(19.0, result.at({0, 0}), 1e-9, "Matmul (0,0) should be 19.0");
    assertNear(22.0, result.at({0, 1}), 1e-9, "Matmul (0,1) should be 22.0");
    assertNear(43.0, result.at({1, 0}), 1e-9, "Matmul (1,0) should be 43.0");
    assertNear(50.0, result.at({1, 1}), 1e-9, "Matmul (1,1) should be 50.0");

    // Test 3x2 * 2x3 multiplication
    std::vector<std::vector<double>> mat3_data = {
        {1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}};
    std::vector<std::vector<double>> mat4_data = {{7.0, 8.0, 9.0},
                                                  {10.0, 11.0, 12.0}};

    NDArray mat3(mat3_data);
    NDArray mat4(mat4_data);

    NDArray result2 = mat3.matmul(mat4);
    assertEqual(size_t(3), result2.shape()[0], "Result should have 3 rows");
    assertEqual(size_t(3), result2.shape()[1], "Result should have 3 columns");

    // Expected: [[1*7+2*10, 1*8+2*11, 1*9+2*12], [3*7+4*10, 3*8+4*11,
    // 3*9+4*12], [5*7+6*10, 5*8+6*11, 5*9+6*12]]
    //         = [[27, 30, 33], [61, 68, 75], [95, 106, 117]]
    assertNear(27.0, result2.at({0, 0}), 1e-9, "Result (0,0) should be 27.0");
    assertNear(30.0, result2.at({0, 1}), 1e-9, "Result (0,1) should be 30.0");
    assertNear(33.0, result2.at({0, 2}), 1e-9, "Result (0,2) should be 33.0");
    assertNear(61.0, result2.at({1, 0}), 1e-9, "Result (1,0) should be 61.0");
    assertNear(117.0, result2.at({2, 2}), 1e-9, "Result (2,2) should be 117.0");
  }
};

/**
 * @class NDArrayErrorTest
 * @brief Test NDArray error conditions
 */
class NDArrayErrorTest : public TestCase {
public:
  NDArrayErrorTest() : TestCase("NDArrayErrorTest") {}

protected:
  void test() override {
    // Test out of bounds access (1D)
    NDArray arr({3});
    assertThrows<std::out_of_range>(
        [&]() {
          arr[5];  // Out of bounds
        },
        "Should throw out_of_range for 1D out of bounds access");

    // Test out of bounds access (multi-dimensional)
    NDArray arr2d({2, 3});
    assertThrows<std::out_of_range>(
        [&]() {
          arr2d.at({3, 1});  // Out of bounds on first dimension
        },
        "Should throw out_of_range for multi-dimensional out of bounds access");

    assertThrows<std::out_of_range>(
        [&]() {
          arr2d.at({1, 5});  // Out of bounds on second dimension
        },
        "Should throw out_of_range for multi-dimensional out of bounds access");

    // Test invalid reshape
    NDArray arr3({6});
    assertThrows<std::invalid_argument>(
        [&]() {
          arr3.reshape({2, 4});  // 2*4=8 != 6
        },
        "Should throw invalid_argument for incompatible reshape");

    // Test incompatible arithmetic operations
    NDArray arr4({2, 3});
    NDArray arr5({3, 2});
    assertThrows<std::invalid_argument>(
        [&]() {
          auto result = arr4 + arr5;  // Incompatible shapes
        },
        "Should throw invalid_argument for incompatible addition");

    // Test incompatible matrix multiplication
    NDArray mat1({2, 3});
    NDArray mat2({2, 4});  // Should be (3, x) for valid multiplication
    assertThrows<std::invalid_argument>(
        [&]() {
          auto result = mat1.matmul(mat2);  // Incompatible shapes
        },
        "Should throw invalid_argument for incompatible matmul");
  }
};

}  // namespace test
}  // namespace MLLib
