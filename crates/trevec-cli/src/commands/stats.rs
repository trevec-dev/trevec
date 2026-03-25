use anyhow::Result;
use std::path::PathBuf;

use trevec_core::model::QueryStats;

use crate::commands::projects::{load_registry, shorten_path};

pub async fn run(path: PathBuf, all: bool, json: bool) -> Result<()> {
    if all {
        run_all(json)
    } else {
        run_single(path, json)
    }
}

fn run_single(path: PathBuf, json: bool) -> Result<()> {
    let path = path.canonicalize().unwrap_or(path);
    let stats_path = path.join(".trevec/stats.json");
    let stats = QueryStats::load(&stats_path);

    if json {
        let output = serde_json::to_string_pretty(&stats)?;
        println!("{output}");
        return Ok(());
    }

    let display_path = shorten_path(&path.to_string_lossy());

    if stats.total_queries == 0 {
        eprintln!("Trevec Stats \u{2014} {display_path}");
        eprintln!();
        eprintln!("  No queries recorded yet.");
        eprintln!("  Run `trevec serve` and use get_context to start tracking.");
        return Ok(());
    }

    eprintln!("Trevec Stats \u{2014} {display_path}");
    eprintln!();
    eprintln!("  Queries served     {}", format_number(stats.total_queries));
    eprintln!(
        "  Tokens returned    {}",
        format_number(stats.total_tokens_returned)
    );
    eprintln!(
        "  Tokens saved       {}  ({:.1}%)",
        format_number(stats.tokens_saved()),
        stats.savings_percentage()
    );
    if stats.total_reindexes > 0 {
        eprintln!(
            "  Reindexes          {}",
            format_number(stats.total_reindexes)
        );
        eprintln!(
            "  Last reindex       {}ms ({} files)",
            format_number(stats.last_reindex_ms),
            format_number(stats.last_reindex_files)
        );
    }

    if let Some(ts) = stats.first_query_at {
        let date = format_date(ts);
        eprintln!();
        eprintln!("  Since: {date}");
    }

    Ok(())
}

fn run_all(json: bool) -> Result<()> {
    let entries = load_registry();

    if entries.is_empty() {
        eprintln!("No tracked projects.");
        eprintln!("Run `trevec init` in a repository to register it.");
        return Ok(());
    }

    // Collect per-project stats
    let mut project_stats: Vec<(String, QueryStats)> = Vec::new();
    let mut total = QueryStats::default();

    for entry in &entries {
        let stats_path = PathBuf::from(&entry.path).join(".trevec/stats.json");
        let stats = QueryStats::load(&stats_path);
        total.merge(&stats);
        project_stats.push((entry.path.clone(), stats));
    }

    if json {
        let projects_json: Vec<serde_json::Value> = project_stats
            .iter()
            .map(|(path, stats)| {
                serde_json::json!({
                    "path": path,
                    "total_queries": stats.total_queries,
                    "total_tokens_returned": stats.total_tokens_returned,
                    "total_source_file_tokens": stats.total_source_file_tokens,
                    "tokens_saved": stats.tokens_saved(),
                    "savings_percentage": stats.savings_percentage(),
                })
            })
            .collect();

        let output = serde_json::json!({
            "projects": projects_json,
            "total": {
                "total_queries": total.total_queries,
                "total_tokens_returned": total.total_tokens_returned,
                "total_source_file_tokens": total.total_source_file_tokens,
                "tokens_saved": total.tokens_saved(),
                "savings_percentage": total.savings_percentage(),
            }
        });
        println!("{}", serde_json::to_string_pretty(&output)?);
        return Ok(());
    }

    eprintln!("Trevec Stats \u{2014} All Projects");
    eprintln!();
    eprintln!(
        "  {:<40} {:>10} {:>16} {:>10}",
        "PROJECT", "QUERIES", "TOKENS SAVED", "SAVINGS"
    );

    for (path, stats) in &project_stats {
        let display = shorten_path(path);
        // Truncate long paths
        let display = if display.len() > 38 {
            format!("..{}", &display[display.len() - 36..])
        } else {
            display
        };

        if stats.total_queries == 0 {
            eprintln!("  {:<40} {:>10} {:>16} {:>10}", display, "\u{2014}", "\u{2014}", "\u{2014}");
        } else {
            eprintln!(
                "  {:<40} {:>10} {:>16} {:>9.1}%",
                display,
                format_number(stats.total_queries),
                format_number(stats.tokens_saved()),
                stats.savings_percentage()
            );
        }
    }

    eprintln!("  {}", "\u{2500}".repeat(80));

    if total.total_queries == 0 {
        eprintln!("  No queries recorded yet across any project.");
    } else {
        eprintln!(
            "  {:<40} {:>10} {:>16} {:>9.1}%",
            "Total",
            format_number(total.total_queries),
            format_number(total.tokens_saved()),
            total.savings_percentage()
        );
    }

    Ok(())
}

fn format_number(n: u64) -> String {
    if n < 1_000 {
        n.to_string()
    } else if n < 1_000_000 {
        format!("{},{:03}", n / 1_000, n % 1_000)
    } else {
        let millions = n / 1_000_000;
        let thousands = (n % 1_000_000) / 1_000;
        let remainder = n % 1_000;
        format!("{},{:03},{:03}", millions, thousands, remainder)
    }
}

fn format_date(unix_ts: i64) -> String {
    // Simple date formatting without chrono dependency
    let secs_per_day: i64 = 86400;
    let days_since_epoch = unix_ts / secs_per_day;

    // Compute year/month/day from days since 1970-01-01
    let mut days = days_since_epoch;
    let mut year = 1970i64;

    loop {
        let days_in_year = if is_leap_year(year) { 366 } else { 365 };
        if days < days_in_year {
            break;
        }
        days -= days_in_year;
        year += 1;
    }

    let month_days = if is_leap_year(year) {
        [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    } else {
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    };

    let mut month = 0;
    for (i, &md) in month_days.iter().enumerate() {
        if days < md {
            month = i + 1;
            break;
        }
        days -= md;
    }

    let day = days + 1;
    format!("{year}-{month:02}-{day:02}")
}

fn is_leap_year(year: i64) -> bool {
    (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_number() {
        assert_eq!(format_number(0), "0");
        assert_eq!(format_number(42), "42");
        assert_eq!(format_number(1_234), "1,234");
        assert_eq!(format_number(1_234_567), "1,234,567");
    }

    #[test]
    fn test_format_date() {
        // 2026-02-28 = 56 years + 14 leap days + 58 days = 20512 days
        // Let's use a known value: 1970-01-01 = 0
        assert_eq!(format_date(0), "1970-01-01");
        // 2000-01-01 = 10957 days * 86400
        assert_eq!(format_date(946684800), "2000-01-01");
    }

    #[test]
    fn test_run_single_no_stats() {
        let tmp = tempfile::tempdir().unwrap();
        let trevec_dir = tmp.path().join(".trevec");
        std::fs::create_dir_all(&trevec_dir).unwrap();
        // No stats.json — should show "no queries"
        let result = run_single(tmp.path().to_path_buf(), false);
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_single_with_stats() {
        let tmp = tempfile::tempdir().unwrap();
        let trevec_dir = tmp.path().join(".trevec");
        std::fs::create_dir_all(&trevec_dir).unwrap();

        let mut stats = QueryStats::default();
        stats.record_query(500, 5000);
        stats.record_query(300, 3000);
        stats.save(&trevec_dir.join("stats.json")).unwrap();

        let result = run_single(tmp.path().to_path_buf(), false);
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_single_json() {
        let tmp = tempfile::tempdir().unwrap();
        let trevec_dir = tmp.path().join(".trevec");
        std::fs::create_dir_all(&trevec_dir).unwrap();

        let mut stats = QueryStats::default();
        stats.record_query(500, 5000);
        stats.save(&trevec_dir.join("stats.json")).unwrap();

        let result = run_single(tmp.path().to_path_buf(), true);
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_single_malformed_stats_json() {
        let tmp = tempfile::tempdir().unwrap();
        let trevec_dir = tmp.path().join(".trevec");
        std::fs::create_dir_all(&trevec_dir).unwrap();
        std::fs::write(trevec_dir.join("stats.json"), "{{invalid json").unwrap();

        // Should not crash — falls back to default empty stats
        let result = run_single(tmp.path().to_path_buf(), false);
        assert!(result.is_ok());
    }

    #[test]
    fn test_format_date_leap_year() {
        // 2024-02-29 is a leap day
        // 2024-01-01 = 19723 days since epoch * 86400 = 1704067200
        // Jan has 31 days, so Feb 29 = 1704067200 + (31 + 28) * 86400 = 1704067200 + 5097600
        // Wait, 2024 is a leap year, so Feb has 29 days. Feb 29 = day 60 (0-indexed 59)
        // 1704067200 + 59 * 86400 = 1704067200 + 5097600 = 1709164800
        assert_eq!(format_date(1709164800), "2024-02-29");
    }

    #[test]
    fn test_format_date_end_of_year() {
        // 2024-12-31 = day 365 of 2024 (leap year), epoch day 20088
        // 20088 * 86400 = 1735603200
        assert_eq!(format_date(1735603200), "2024-12-31");
    }

    #[test]
    fn test_format_number_edge_cases() {
        assert_eq!(format_number(999), "999");
        assert_eq!(format_number(1000), "1,000");
        assert_eq!(format_number(999_999), "999,999");
        assert_eq!(format_number(1_000_000), "1,000,000");
    }
}
