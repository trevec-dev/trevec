# Contributing to Trevec

Thanks for your interest in contributing! We're a young project and appreciate all help — bug reports, feature requests, docs improvements, and code contributions.

## Getting Started

1. Fork the repo and clone it
2. Install Rust 1.91+ (pinned in `rust-toolchain.toml`)
3. Build: `cargo build`
4. Run tests: `cargo test --workspace`

## Ways to Contribute

- **Bug reports** — Open an issue with steps to reproduce
- **Feature requests** — Open an issue describing the use case
- **Code** — Pick an open issue, comment that you're working on it, and open a PR
- **Docs** — Fix typos, improve explanations, add examples

## Pull Requests

1. Create a branch from `main`
2. Make your changes
3. Run `cargo test --workspace` and `cargo clippy --workspace` before submitting
4. Open a PR with a clear description of what you changed and why
5. Keep PRs focused — one feature or fix per PR

## Code Style

- Follow existing patterns in the codebase
- Run `cargo fmt` before committing
- No warnings from `cargo clippy`

## Reporting Issues

When filing a bug, please include:
- Trevec version (`trevec --version`)
- OS and architecture
- Steps to reproduce
- Expected vs actual behavior

## License

By contributing, you agree that your contributions will be licensed under the [Apache 2.0 License](LICENSE).
