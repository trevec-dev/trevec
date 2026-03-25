/**
 * Utility functions for data processing and validation.
 */

/**
 * Validate an email address format.
 * @param {string} email - The email to validate
 * @returns {boolean} True if the email is valid
 */
function validateEmail(email) {
  const regex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return regex.test(email);
}

/**
 * Sanitize user input to prevent XSS.
 * @param {string} input - The raw input
 * @returns {string} Sanitized string
 */
function sanitizeInput(input) {
  return input
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

/**
 * Generate a random token for session management.
 * @param {number} length - Token length
 * @returns {string} Random token
 */
function generateToken(length = 32) {
  const chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
  let result = "";
  for (let i = 0; i < length; i++) {
    result += chars.charAt(Math.floor(Math.random() * chars.length));
  }
  return result;
}

class DataProcessor {
  constructor(options = {}) {
    this.batchSize = options.batchSize || 100;
    this.delimiter = options.delimiter || ",";
  }

  /**
   * Process a batch of records.
   * @param {Array} records - Records to process
   * @returns {Array} Processed records
   */
  process(records) {
    const batches = this.chunk(records, this.batchSize);
    return batches.flatMap((batch) => this.processBatch(batch));
  }

  processBatch(batch) {
    return batch.map((record) => ({
      ...record,
      processed: true,
      timestamp: Date.now(),
    }));
  }

  chunk(array, size) {
    const chunks = [];
    for (let i = 0; i < array.length; i += size) {
      chunks.push(array.slice(i, i + size));
    }
    return chunks;
  }
}

module.exports = { validateEmail, sanitizeInput, generateToken, DataProcessor };
