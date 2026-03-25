export interface MemoryResult {
  id: string;
  memory: string;
  userId?: string;
  role?: string;
  createdAt?: number;
  score: number;
}

export interface IndexStats {
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
}

export class Trevec {
  constructor(project?: string);
  static forRepo(repoPath?: string): Trevec;

  add(content: string, userId: string, metadata?: Record<string, string>): string;
  search(query: string, userId?: string, limit?: number): MemoryResult[];
  getAll(userId: string): MemoryResult[];
  delete(memoryId: string): boolean;
  deleteAll(userId: string): number;

  index(repoPath?: string): IndexStats;
  query(queryText: string, budget?: number): QueryResult;

  get nodeCount(): number;
  get isIndexed(): boolean;
  toString(): string;
}
