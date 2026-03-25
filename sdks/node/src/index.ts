/**
 * Trevec — Universal Context Graph for AI Agents
 *
 * @example
 * ```typescript
 * import { Trevec } from 'trevec';
 *
 * const tv = new Trevec();
 * tv.add("I love hiking in Denver", { userId: "alex" });
 * const results = tv.search("hobbies", { userId: "alex" });
 * ```
 */

import * as fs from "fs";
import * as path from "path";
import * as crypto from "crypto";
import { execSync } from "child_process";

// ── Types ────────────────────────────────────────────────────────────────────

export interface TrevecOptions {
  /** Project name for persistent storage. Omit for ephemeral in-memory mode. */
  project?: string;
  /** Explicit data directory path. Overrides project-based path. */
  dataDir?: string;
  /** Path to a code repository for indexing. */
  repoPath?: string;
}

export interface IndexStats {
  filesDiscovered: number;
  filesParsed: number;
  nodesExtracted: number;
  edgesBuilt: number;
  totalMs: number;
}

export interface QueryResult {
  query: string;
  context: string;
  totalTokens: number;
  retrievalMs: number;
  nodes: QueryNode[];
}

export interface QueryNode {
  filePath: string;
  name: string;
  kind: string;
  signature: string;
  source: string;
  startLine: number;
  endLine: number;
}

export interface AddOptions {
  /** User ID for scoping memories. */
  userId: string;
  /** Optional metadata key-value pairs. */
  metadata?: Record<string, string>;
}

export interface SearchOptions {
  /** User ID to scope search to. Omit to search all users. */
  userId?: string;
  /** Maximum results to return. Default: 10. */
  limit?: number;
}

export interface MemoryResult {
  id: string;
  memory: string;
  userId?: string;
  role?: string;
  createdAt?: number;
  score: number;
}

export interface Message {
  role: string;
  content: string;
}

// ── Internal Types ───────────────────────────────────────────────────────────

interface MemoryNode {
  id: string;
  content: string;
  userId: string;
  role?: string;
  metadata: Record<string, string>;
  createdAt: number;
  searchText: string;
}

// ── Trevec Class ─────────────────────────────────────────────────────────────

export class Trevec {
  private nodes: Map<string, MemoryNode> = new Map();
  private dataDir: string | null = null;
  private project: string | null = null;
  private repoPath: string | null = null;

  /**
   * Create a new Trevec instance.
   *
   * @example
   * ```typescript
   * // Ephemeral (in-memory)
   * const tv = new Trevec();
   *
   * // Persistent
   * const tv = new Trevec({ project: "my-app" });
   * ```
   */
  constructor(options?: TrevecOptions | string) {
    if (typeof options === "string") {
      // Trevec("my-project") shorthand
      this.project = options;
      this.dataDir = this.resolveDataDir(options);
    } else if (options?.project) {
      this.project = options.project;
      this.dataDir = options.dataDir || this.resolveDataDir(options.project);
    } else if (options?.dataDir) {
      this.dataDir = options.dataDir;
    }

    if (typeof options === "object" && options?.repoPath) {
      this.repoPath = options.repoPath;
    }

    if (this.dataDir) {
      fs.mkdirSync(this.dataDir, { recursive: true });
      this.loadFromDisk();
    }
  }

  // ── Core API ─────────────────────────────────────────────────────────────

  /**
   * Add a memory. Accepts a string or an array of messages.
   *
   * @example
   * ```typescript
   * tv.add("I love hiking", { userId: "alex" });
   *
   * tv.add([
   *   { role: "user", content: "What is calculus?" },
   *   { role: "assistant", content: "The study of change..." },
   * ], { userId: "student01" });
   * ```
   */
  add(content: string | Message[], options: AddOptions): string | string[] {
    if (Array.isArray(content)) {
      return this.addMessages(content, options);
    }
    return this.addSingle(content, options);
  }

  /**
   * Search memories. Scoped by userId if provided.
   *
   * @example
   * ```typescript
   * const results = tv.search("hobbies", { userId: "alex" });
   * ```
   */
  search(query: string, options?: SearchOptions): MemoryResult[] {
    const limit = options?.limit ?? 10;
    const userId = options?.userId;
    const queryTerms = query.toLowerCase().split(/\s+/);

    const scored: Array<{ node: MemoryNode; score: number }> = [];

    for (const node of this.nodes.values()) {
      // User scoping
      if (userId && node.userId !== userId) continue;

      const text = node.searchText.toLowerCase();
      let score = 0;

      for (const term of queryTerms) {
        if (text.includes(term)) score += 1;
        if (node.content.toLowerCase().includes(term)) score += 2;
      }

      if (score > 0) {
        scored.push({ node, score });
      }
    }

    scored.sort((a, b) => b.score - a.score);

    return scored.slice(0, limit).map(({ node, score }) => ({
      id: node.id,
      memory: node.content.length > 200 ? node.content.slice(0, 200) + "..." : node.content,
      userId: node.userId,
      role: node.role,
      createdAt: node.createdAt,
      score,
    }));
  }

  /**
   * Get all memories for a user.
   */
  getAll(userId: string): MemoryResult[] {
    const results: MemoryResult[] = [];
    for (const node of this.nodes.values()) {
      if (node.userId === userId) {
        results.push({
          id: node.id,
          memory: node.content.length > 200 ? node.content.slice(0, 200) + "..." : node.content,
          userId: node.userId,
          role: node.role,
          createdAt: node.createdAt,
          score: 0,
        });
      }
    }
    return results;
  }

  /**
   * Delete a specific memory by ID.
   */
  delete(memoryId: string): boolean {
    const deleted = this.nodes.delete(memoryId);
    if (deleted) this.saveToDisk();
    return deleted;
  }

  /**
   * Delete all memories for a user. Returns the number deleted.
   */
  deleteAll(userId: string): number {
    const toDelete: string[] = [];
    for (const [id, node] of this.nodes) {
      if (node.userId === userId) toDelete.push(id);
    }
    for (const id of toDelete) {
      this.nodes.delete(id);
    }
    if (toDelete.length > 0) this.saveToDisk();
    return toDelete.length;
  }

  /**
   * Total number of memories stored.
   */
  get nodeCount(): number {
    return this.nodes.size;
  }

  // ── Code Context (requires trevec CLI installed) ─────────────────────────

  /**
   * Create a Trevec instance for a code repository.
   *
   * @example
   * ```typescript
   * const tv = Trevec.forRepo("/path/to/repo");
   * const stats = tv.index();
   * const result = tv.query("How does auth work?");
   * ```
   */
  static forRepo(repoPath: string): Trevec {
    return new Trevec({ repoPath, project: path.basename(repoPath) });
  }

  /**
   * Index a code repository using Tree-sitter.
   * Requires the trevec CLI to be installed.
   *
   * @example
   * ```typescript
   * const stats = tv.index();
   * console.log(`Indexed ${stats.nodesExtracted} nodes`);
   * ```
   */
  index(repoPath?: string): IndexStats {
    const repo = repoPath || this.repoPath;
    if (!repo) {
      throw new Error("No repo path. Use Trevec.forRepo('/path') or pass a path to index().");
    }
    this.repoPath = repo;

    this.ensureCli();
    const dataDir = this.dataDir || path.join(repo, ".trevec");

    try {
      const output = execSync(
        `trevec index "${repo}" --data-dir "${dataDir}" --json`,
        { encoding: "utf-8", timeout: 300_000, stdio: ["pipe", "pipe", "pipe"] }
      );

      try {
        const stats = JSON.parse(output.trim());
        return {
          filesDiscovered: stats.files_discovered || 0,
          filesParsed: stats.files_parsed || 0,
          nodesExtracted: stats.nodes_extracted || 0,
          edgesBuilt: stats.edges_built || 0,
          totalMs: stats.total_ms || 0,
        };
      } catch {
        // CLI may not output JSON, parse from text
        return { filesDiscovered: 0, filesParsed: 0, nodesExtracted: 0, edgesBuilt: 0, totalMs: 0 };
      }
    } catch (e: unknown) {
      const err = e as Error;
      throw new Error(`Index failed: ${err.message}`);
    }
  }

  /**
   * Query the indexed codebase with hybrid search.
   * Requires index() to have been called first.
   *
   * @example
   * ```typescript
   * const result = tv.query("How does authentication work?");
   * console.log(result.context);
   * ```
   */
  query(queryText: string, options?: { budget?: number }): QueryResult {
    const repo = this.repoPath;
    if (!repo) {
      throw new Error("No repo path. Use Trevec.forRepo('/path') first.");
    }

    this.ensureCli();
    const dataDir = this.dataDir || path.join(repo, ".trevec");
    const budget = options?.budget || 4096;

    try {
      const output = execSync(
        `trevec ask "${queryText}" --path "${repo}" --data-dir "${dataDir}" --budget ${budget} --json`,
        { encoding: "utf-8", timeout: 60_000, stdio: ["pipe", "pipe", "pipe"] }
      );

      try {
        const bundle = JSON.parse(output.trim());
        return {
          query: bundle.query || queryText,
          context: bundle.formatted || output,
          totalTokens: bundle.total_estimated_tokens || 0,
          retrievalMs: bundle.retrieval_ms || 0,
          nodes: (bundle.included_nodes || []).map((n: Record<string, unknown>) => ({
            filePath: n.file_path || "",
            name: n.name || "",
            kind: n.kind || "",
            signature: n.signature || "",
            source: n.source_text || "",
            startLine: ((n.span as Record<string, number>)?.start_line || 0) + 1,
            endLine: ((n.span as Record<string, number>)?.end_line || 0) + 1,
          })),
        };
      } catch {
        return {
          query: queryText,
          context: output,
          totalTokens: 0,
          retrievalMs: 0,
          nodes: [],
        };
      }
    } catch (e: unknown) {
      const err = e as Error;
      throw new Error(`Query failed: ${err.message}`);
    }
  }

  /**
   * Whether the trevec CLI is installed.
   */
  get cliAvailable(): boolean {
    try {
      execSync("trevec --version", { stdio: "pipe" });
      return true;
    } catch {
      return false;
    }
  }

  /**
   * String representation.
   */
  toString(): string {
    return `Trevec(project='${this.project || "ephemeral"}', nodes=${this.nodes.size})`;
  }

  // ── Private Methods ──────────────────────────────────────────────────────

  private addSingle(content: string, options: AddOptions): string {
    const now = Date.now();
    const id = crypto
      .createHash("sha256")
      .update(`${options.userId}:${now}:${content.slice(0, 100)}`)
      .digest("hex")
      .slice(0, 32);

    const node: MemoryNode = {
      id,
      content,
      userId: options.userId,
      role: options.metadata?.role,
      metadata: options.metadata || {},
      createdAt: Math.floor(now / 1000),
      searchText: `${options.userId} ${content}`,
    };

    this.nodes.set(id, node);
    this.saveToDisk();
    return id;
  }

  private addMessages(messages: Message[], options: AddOptions): string[] {
    const ids: string[] = [];
    for (const msg of messages) {
      if (!msg.content) continue;
      const meta = { ...options.metadata, role: msg.role };
      const id = this.addSingle(msg.content, { ...options, metadata: meta });
      ids.push(id);
    }
    return ids;
  }

  private ensureCli(): void {
    try {
      execSync("trevec --version", { stdio: "pipe" });
    } catch {
      throw new Error(
        "trevec CLI not found. Install it: curl -fsSL dl.trevec.dev/install.sh | sh\n" +
        "Code indexing requires the CLI. Memory features (add/search) work without it."
      );
    }
  }

  private resolveDataDir(project: string): string {
    const home = process.env.HOME || process.env.USERPROFILE || "/tmp";
    return path.join(home, ".trevec", project);
  }

  private saveToDisk(): void {
    if (!this.dataDir) return;
    try {
      const data = Array.from(this.nodes.values());
      const filePath = path.join(this.dataDir, "memories.json");
      fs.writeFileSync(filePath, JSON.stringify(data, null, 2));
    } catch {
      // Silent fail for ephemeral mode
    }
  }

  private loadFromDisk(): void {
    if (!this.dataDir) return;
    try {
      const filePath = path.join(this.dataDir, "memories.json");
      if (fs.existsSync(filePath)) {
        const data: MemoryNode[] = JSON.parse(fs.readFileSync(filePath, "utf-8"));
        for (const node of data) {
          this.nodes.set(node.id, node);
        }
      }
    } catch {
      // Start fresh if file is corrupted
    }
  }
}

export default Trevec;
