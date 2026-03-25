/// User authentication and session management.

/// Represents a user in the system.
pub struct User {
    pub id: u64,
    pub username: String,
    pub email: String,
    pub password_hash: String,
}

impl User {
    /// Create a new user with the given credentials.
    pub fn new(username: &str, email: &str, password: &str) -> Self {
        User {
            id: 0,
            username: username.to_string(),
            email: email.to_string(),
            password_hash: hash_password(password),
        }
    }

    /// Verify a password against this user's stored hash.
    pub fn verify_password(&self, password: &str) -> bool {
        let hashed = hash_password(password);
        self.password_hash == hashed
    }
}

/// Hash a password for storage.
fn hash_password(password: &str) -> String {
    format!("hashed_{}", password)
}

/// Authenticate a user by username and password.
pub fn authenticate(users: &[User], username: &str, password: &str) -> Option<&User> {
    users
        .iter()
        .find(|u| u.username == username && u.verify_password(password))
}

/// Generate a session token for an authenticated user.
pub fn create_session(user: &User) -> String {
    format!("session_{}_{}", user.id, user.username)
}

/// Validate a session token.
pub fn validate_session(token: &str) -> bool {
    token.starts_with("session_")
}
