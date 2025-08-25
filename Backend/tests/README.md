# Backend Testing Guide

This guide covers the comprehensive testing setup for the Drishtiksha Backend service, including unit tests, integration tests, and end-to-end tests with real server communication.

## 📋 Overview

Our testing strategy includes:

-   **Unit Tests**: Fast, isolated tests for individual components
-   **Integration Tests**: Tests that verify Backend-Server communication using real API keys
-   **End-to-End Tests**: Complete workflow tests with media upload and processing
-   **Performance Tests**: Long-running tests to validate timeout handling and resource management

## 🛠️ Test Setup

### Prerequisites

1. **Environment Configuration**: Ensure your `.env` file is properly configured:

    ```bash
    # Required for integration tests
    SERVER_URL="http://localhost:8000"
    SERVER_API_KEY="your_actual_api_key_here"
    DATABASE_URL="postgres://username:password@localhost:5432/database"
    JWT_SECRET="your_jwt_secret"
    JWT_REFRESH_SECRET="your_jwt_refresh_secret"

    # Optional but recommended
    REDIS_URL="redis://localhost:6379"
    STORAGE_PROVIDER="local"  # Use local storage for tests
    ```

2. **Test Fixtures**: Ensure test media files exist:

    ```text
    tests/fixtures/
    ├── test-video.mp4  # Small video file for testing
    └── test-audio.mp3  # Small audio file for testing
    ```

3. **Services Running**:
    - **Database**: PostgreSQL instance running and accessible
    - **ML Server**: Python server running on port 8000 (for integration tests)
    - **Redis**: Redis instance running (optional, for queue tests)

## 🧪 Running Tests

### Quick Test Commands

```bash
# Run all tests
npm test

# Run only unit tests (fast)
npm run test:unit

# Run integration tests (requires server)
npm run test:integration

# Run end-to-end tests (long-running)
npm run test:e2e

# Run tests in watch mode
npm run test:watch

# Run tests with coverage
npm run test:coverage
```

### Integration Test Runner

For comprehensive integration testing with pre-flight checks:

```bash
# Run the integration test runner (recommended)
npm run test:integration

# Or run specific integration test patterns
node scripts/run-integration-tests.js backend-server
```

The integration test runner will:

1. ✅ Validate environment configuration
2. 🔗 Check server connectivity
3. 🗄️ Verify database connection
4. 📁 Confirm test fixtures exist
5. 🚀 Run integration tests with proper setup

## 📂 Test Structure

```text
tests/
├── fixtures/                          # Test media files
│   ├── test-video.mp4
│   └── test-audio.mp3
├── setup/                             # Test configuration
│   ├── testEnv.js                     # Environment setup and validation
│   └── testSetup.js                   # Global test utilities
├── unit/                              # Unit tests (fast)
│   ├── services/
│   ├── repositories/
│   └── utils/
├── integration/                       # Integration tests (medium)
│   ├── backend-server.integration.test.js
│   └── example.integration.test.js
└── *.e2e.test.js                     # End-to-end tests (slow)
```

## 🔧 Test Configuration

### Jest Configuration

Our Jest setup supports multiple test types with different timeouts:

-   **Unit Tests**: 30 seconds timeout
-   **Integration Tests**: 15 minutes timeout
-   **E2E Tests**: 15 minutes timeout

### Environment Variables

The test environment automatically configures:

-   `NODE_ENV=test`
-   `STORAGE_PROVIDER=local` (avoids cloud storage costs)
-   `TEST_TYPE=integration` (for integration test runs)

## 📝 Writing Tests

### Integration Test Example

```javascript
import { TEST_CONFIG, testUtils } from "../setup/testEnv.js";
import { app } from "../../src/app.js";
import request from "supertest";

describe("My Integration Test", () => {
    let testUser, authToken;

    beforeAll(async () => {
        // Create test user with utilities
        const userData = testUtils.createTestUserData();
        // ... setup logic
    });

    afterAll(async () => {
        // Cleanup test data
        // ... cleanup logic
    });

    it("should test server integration", async () => {
        if (TEST_CONFIG.skipServerTests) {
            console.log("Skipping - server not available");
            return;
        }

        // Your test logic with real server communication
        const result = await testUtils.retry(async () => {
            return await someServerOperation();
        });

        expect(result).toBeDefined();
    });
});
```

### Best Practices

1. **Use Real Environment**: Integration tests use actual `.env` configuration
2. **Proper Cleanup**: Always clean up test users and data in `afterAll`
3. **Timeout Handling**: Use appropriate timeouts for long-running operations
4. **Retry Logic**: Use `testUtils.retry()` for potentially flaky operations
5. **Skip Gracefully**: Check `TEST_CONFIG.skipServerTests` when server is unavailable
6. **Descriptive Logging**: Add console.log statements for test progress tracking

## 🔍 Test Utilities

### Available Utilities

```javascript
import { testUtils } from "../setup/testEnv.js";

// Generate unique test identifiers
const id = testUtils.generateTestId();

// Create test user data
const userData = testUtils.createTestUserData();

// Wait for specified duration
await testUtils.wait(5000); // 5 seconds

// Retry operations with exponential backoff
const result = await testUtils.retry(
    async () => {
        return await potentiallyFlakyOperation();
    },
    3,
    1000
); // 3 attempts, starting with 1 second delay
```

### Test Configuration

```javascript
import TEST_CONFIG from "../setup/testEnv.js";

// Check configuration
if (TEST_CONFIG.skipServerTests) {
    console.log("Server tests disabled");
}

// Use configured values
const serverUrl = TEST_CONFIG.serverUrl;
const maxTimeout = TEST_CONFIG.maxPollingTimeout;
```

## 🐛 Debugging Tests

### Common Issues

1. **Server Connection Failed**:

    ```bash
    # Make sure the Python server is running
    cd ../Server
    python main.py
    ```

2. **Database Connection Failed**:

    ```bash
    # Check DATABASE_URL in .env
    # Ensure PostgreSQL is running
    ```

3. **Test Fixtures Missing**:

    ```bash
    # Add test media files to tests/fixtures/
    # Or some tests will be skipped
    ```

4. **Timeout Issues**:

    ```bash
    # For long-running tests, use integration runner
    npm run test:integration

    # Or increase timeout in jest.config.json
    ```

### Verbose Logging

Enable detailed test logging:

```bash
# Run with verbose output
npm test -- --verbose

# Run integration tests with detailed logs
NODE_ENV=test DEBUG=* npm run test:integration
```

## 📊 Test Coverage

Generate test coverage reports:

```bash
npm run test:coverage
```

Coverage reports will be generated in the `coverage/` directory.

## 🚀 Continuous Integration

For CI environments, use the integration test runner which includes pre-flight checks:

```bash
# In CI pipeline
npm run test:unit      # Fast unit tests first
npm run test:integration  # Then integration tests if server available
```

## ⚡ Performance Considerations

-   **Unit Tests**: Should complete in under 30 seconds
-   **Integration Tests**: May take up to 15 minutes due to ML processing
-   **Parallel Execution**: Integration tests run serially to avoid conflicts
-   **Resource Cleanup**: Tests automatically clean up created resources

## 🔐 Security Notes

-   Test users are automatically created and destroyed
-   Real API keys are used but only with test data
-   Local storage is used by default to avoid cloud storage costs
-   Database operations are isolated per test run

---

For questions or issues with the testing setup, please refer to the test configuration files or create an issue in the project repository.
