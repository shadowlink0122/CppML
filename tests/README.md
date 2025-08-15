# MLLib Testing Framework

## Overview

MLLib includes a comprehensive testing framework with both unit tests and integration tests. The testing system features output capture to keep test results clean and provides detailed reporting of test outcomes.

## Test Structure

```
tests/
├── test_main.cpp              # Main test runner
├── common/                    # Shared test utilities
│   ├── test_utils.hpp         # Test framework and output capture
│   └── test_utils.cpp         # Test utility implementations
├── unit/                      # Unit tests (mirrors include/MLLib structure)
│   ├── unit_test_main.cpp     # Unit test main entry point
│   ├── MLLib/
│   │   ├── test_config.hpp    # Config module tests
│   │   ├── test_ndarray.hpp   # NDArray tests
│   │   ├── layer/
│   │   │   ├── test_dense.hpp # Dense layer tests
│   │   │   └── activation/
│   │   │       └── test_activation.hpp # Activation function tests
│   │   ├── model/
│   │   │   ├── test_sequential.hpp    # Sequential model tests
│   │   │   └── test_model_io.hpp      # Model I/O tests
│   │   ├── loss/              # Loss function tests (future)
│   │   ├── optimizer/         # Optimizer tests (future)
│   │   └── util/              # Utility tests (future)
└── integration/               # Integration tests
    └── integration_test_main.cpp # End-to-end integration tests
```

## Running Tests

### All Tests
```bash
make test           # Run both unit and integration tests
make test-all       # Run comprehensive test runner
```

### Specific Test Types
```bash
make unit-test      # Run only unit tests
make integration-test # Run only integration tests
```

### Manual Test Execution
```bash
# Build and run unit tests manually
./build/tests/unit_tests

# Build and run integration tests manually
./build/tests/integration_tests

# Run test runner with options
./build/tests/test_runner --unit-only
./build/tests/test_runner --integration-only
./build/tests/test_runner --help
```

## Test Features

### Output Capture
The testing framework includes sophisticated output capture that:
- Prevents test output from interfering with test reporting
- Captures both stdout and stderr during test execution
- Provides clean, structured test results
- Works for both unit tests and integration tests

### Test Organization
- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test component interactions and end-to-end workflows
- **Error Testing**: Comprehensive error condition testing
- **Performance Tests**: Basic performance and stability verification

### Test Categories

#### Unit Tests
1. **Config Tests**: Configuration constants and mathematical properties
2. **NDArray Tests**: Multi-dimensional array operations and arithmetic
3. **Dense Layer Tests**: Fully connected layer functionality
4. **Activation Tests**: ReLU, Sigmoid, and Tanh activation functions
5. **Sequential Model Tests**: Model construction and training
6. **Model I/O Tests**: Saving and loading in multiple formats

#### Integration Tests
1. **XOR Problem**: End-to-end training and prediction
2. **Model I/O Workflow**: Complete save/load/predict cycles
3. **Multi-Layer Architecture**: Complex model testing
4. **Performance and Stability**: Large dataset handling

## Test Utilities

### TestCase Base Class
```cpp
class MyTest : public MLLib::test::TestCase {
public:
    MyTest() : TestCase("MyTest") {}
    
protected:
    void test() override {
        // Test implementation
        assertEqual(expected, actual, "Description");
        assertTrue(condition, "Description");
        assertNear(expected, actual, tolerance, "Description");
        // ... more assertions
    }
};
```

### Available Assertions
- `assertTrue(condition, message)`
- `assertFalse(condition, message)`
- `assertEqual(expected, actual, message)`
- `assertNotEqual(not_expected, actual, message)`
- `assertNear(expected, actual, tolerance, message)`
- `assertNotNull(ptr, message)`
- `assertNull(ptr, message)`
- `assertVectorEqual(expected, actual, message)`
- `assertVectorNear(expected, actual, tolerance, message)`
- `assertThrows<ExceptionType>(func, message)`
- `assertNoThrow(func, message)`

### Output Capture Usage
```cpp
// Automatic output capture in tests
void test() override {
    OutputCapture capture; // Captures output automatically
    
    // Any output here is captured and doesn't appear in test results
    std::cout << "This won't appear in test output" << std::endl;
    
    // Test assertions work normally
    assertTrue(some_condition, "Test description");
}
```

### Temporary File Utilities
```cpp
// Create temporary files for testing
std::string temp_file = createTempFile("file content");
std::string temp_dir = createTempDirectory();

// Cleanup is automatic, but can be done manually
removeTempFile(temp_file);
removeTempDirectory(temp_dir);
```

## Test Coverage

The test suite covers:

### Core Components
- [x] Configuration management
- [x] Multi-dimensional arrays (NDArray)
- [x] Dense layers
- [x] Activation functions (ReLU, Sigmoid, Tanh)
- [x] Sequential models
- [x] Model I/O (Binary, JSON, Config formats)

### Functionality Areas
- [x] Construction and initialization
- [x] Forward propagation
- [x] Backward propagation (where applicable)
- [x] Parameter management
- [x] Error conditions and edge cases
- [x] Numerical accuracy
- [x] Memory management
- [x] File I/O operations

### Integration Scenarios
- [x] End-to-end training workflows
- [x] Model persistence and loading
- [x] Complex architectures
- [x] Performance and stability
- [x] Cross-component interactions

## Adding New Tests

### Creating Unit Tests

1. Create a new header file in `tests/unit/`:
```cpp
#pragma once
#include "test_utils.hpp"
#include "../../include/MLLib/your_component.hpp"

namespace MLLib {
namespace test {

class YourComponentTest : public TestCase {
public:
    YourComponentTest() : TestCase("YourComponentTest") {}
    
protected:
    void test() override {
        // Your test implementation
    }
};

} // namespace test
} // namespace MLLib
```

2. Add the test to `unit_test_main.cpp`:
```cpp
#include "test_your_component.hpp"

// In main function:
TestSuite your_suite("Your Component Tests");
your_suite.addTest(std::make_unique<YourComponentTest>());
bool suite_result = your_suite.runAll();
```

### Creating Integration Tests

Add integration test classes to `integration_test_main.cpp` following the same pattern as existing integration tests.

## Best Practices

### Test Design
- Write tests for both success and failure cases
- Test edge cases and boundary conditions
- Use descriptive test names and assertion messages
- Keep tests focused and atomic
- Use appropriate tolerances for floating-point comparisons

### Output Management
- Let the output capture system handle console output
- Use assertion messages to provide context for failures
- Structure test output for readability

### Resource Management
- Use temporary files/directories for file I/O tests
- Clean up resources (automatic cleanup is provided)
- Test both normal and error cleanup paths

### Performance
- Keep unit tests fast and focused
- Use integration tests for end-to-end performance testing
- Avoid excessive output in performance tests

## Troubleshooting

### Common Issues

1. **Compilation Errors**:
   - Ensure all required headers are included
   - Check that the library builds successfully first
   - Verify test file paths and includes

2. **Test Failures**:
   - Check assertion messages for details
   - Verify expected vs actual values
   - Consider floating-point precision issues

3. **Output Issues**:
   - Output capture should handle most cases automatically
   - For debugging, temporarily disable output capture
   - Check that test output doesn't interfere with assertions

4. **File I/O Test Issues**:
   - Verify temporary directory creation
   - Check file permissions
   - Ensure cleanup occurs even on test failure

### Debugging Tests

To debug specific tests:
1. Build tests with debug flags: `make debug unit-test`
2. Run specific test suites by modifying the main function
3. Add temporary output (will be captured but visible in debugger)
4. Use assertions with detailed messages

## Continuous Integration

The test framework is designed to work well with CI systems:
- Exit codes indicate success (0) or failure (non-zero)
- Structured output for easy parsing
- Separate unit and integration test runs
- Performance benchmarks for regression detection

## Future Enhancements

Planned improvements:
- [ ] Code coverage reporting
- [ ] Performance regression detection
- [ ] Parallel test execution
- [ ] XML/JSON test result output
- [ ] Memory leak detection integration
- [ ] Automated benchmark reporting
