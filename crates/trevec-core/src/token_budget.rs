/// A token budget tracker that controls how much context to include.
#[derive(Debug, Clone)]
pub struct TokenBudget {
    total: usize,
    used: usize,
}

impl TokenBudget {
    pub fn new(total: usize) -> Self {
        Self { total, used: 0 }
    }

    pub fn total(&self) -> usize {
        self.total
    }

    pub fn used(&self) -> usize {
        self.used
    }

    pub fn remaining(&self) -> usize {
        self.total.saturating_sub(self.used)
    }

    /// Try to consume tokens. Returns true if successful, false if over budget.
    pub fn try_consume(&mut self, tokens: usize) -> bool {
        if self.used + tokens <= self.total {
            self.used += tokens;
            true
        } else {
            false
        }
    }

    pub fn is_exhausted(&self) -> bool {
        self.used >= self.total
    }
}

/// Estimate tokens from byte length (byte_length / 4).
pub fn estimate_tokens(byte_length: usize) -> usize {
    byte_length / 4
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_budget_basic() {
        let mut budget = TokenBudget::new(1000);
        assert_eq!(budget.remaining(), 1000);
        assert!(!budget.is_exhausted());

        assert!(budget.try_consume(500));
        assert_eq!(budget.remaining(), 500);
        assert_eq!(budget.used(), 500);

        assert!(budget.try_consume(500));
        assert_eq!(budget.remaining(), 0);
        assert!(budget.is_exhausted());
    }

    #[test]
    fn test_budget_overflow() {
        let mut budget = TokenBudget::new(100);
        assert!(budget.try_consume(50));
        assert!(!budget.try_consume(60)); // would exceed
        assert_eq!(budget.used(), 50); // unchanged
    }

    #[test]
    fn test_estimate_tokens() {
        assert_eq!(estimate_tokens(400), 100);
        assert_eq!(estimate_tokens(0), 0);
        assert_eq!(estimate_tokens(3), 0); // rounds down
    }
}
