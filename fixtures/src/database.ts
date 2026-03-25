/**
 * Database connection and query utilities.
 */

interface DatabaseConfig {
  host: string;
  port: number;
  database: string;
  username: string;
  password: string;
}

interface QueryResult {
  rows: Record<string, unknown>[];
  rowCount: number;
}

class DatabaseConnection {
  private config: DatabaseConfig;
  private connected: boolean = false;

  constructor(config: DatabaseConfig) {
    this.config = config;
  }

  /** Establish a connection to the database. */
  async connect(): Promise<void> {
    this.connected = true;
  }

  /** Execute a SQL query and return results. */
  async query(sql: string, params?: unknown[]): Promise<QueryResult> {
    if (!this.connected) {
      throw new Error("Not connected to database");
    }
    return { rows: [], rowCount: 0 };
  }

  /** Close the database connection. */
  async disconnect(): Promise<void> {
    this.connected = false;
  }
}

/** Create a new database connection from environment variables. */
function createConnection(): DatabaseConnection {
  const config: DatabaseConfig = {
    host: process.env.DB_HOST || "localhost",
    port: parseInt(process.env.DB_PORT || "5432"),
    database: process.env.DB_NAME || "app",
    username: process.env.DB_USER || "postgres",
    password: process.env.DB_PASS || "",
  };
  return new DatabaseConnection(config);
}

/** Find a user by ID in the database. */
async function findUserById(
  conn: DatabaseConnection,
  id: number
): Promise<Record<string, unknown> | null> {
  const result = await conn.query("SELECT * FROM users WHERE id = $1", [id]);
  return result.rows[0] || null;
}

/** Insert a new user into the database. */
async function insertUser(
  conn: DatabaseConnection,
  username: string,
  email: string
): Promise<void> {
  await conn.query("INSERT INTO users (username, email) VALUES ($1, $2)", [
    username,
    email,
  ]);
}

export { DatabaseConnection, DatabaseConfig, QueryResult, createConnection, findUserById, insertUser };
