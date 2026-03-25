/**
 * Trevec SDK entry point.
 *
 * Tries to load native Rust bindings first (fastest).
 * Falls back to pure TypeScript implementation (works everywhere).
 */

// Try native first
let NativeTrevec: any = null;
try {
  const { loadNative } = require("../native/loader");
  const native = loadNative();
  if (native?.Trevec) {
    NativeTrevec = native.Trevec;
  }
} catch {
  // Native not available, use TypeScript
}

// Re-export types (always from TypeScript for type definitions)
export type {
  TrevecOptions,
  AddOptions,
  SearchOptions,
  MemoryResult,
  Message,
  IndexStats,
  QueryResult,
  QueryNode,
} from "./index";

// Export the best available implementation
import { Trevec as TSTrevec } from "./index";

export const Trevec = NativeTrevec || TSTrevec;
export const isNative = NativeTrevec !== null;
export default Trevec;
