//! Document Parser: extracts structural elements (headings, sections, paragraphs,
//! citations) from Markdown, HTML, and plain text files as UniversalNodes.
//!
//! Produces Section, Fact, and Citation nodes with Contain and Reference edges,
//! all tagged with DomainTag::Document.

use anyhow::Result;
use regex::Regex;
use std::collections::HashMap;
use std::sync::OnceLock;

use trevec_core::universal::*;
use trevec_core::{Confidence, TrevecConfig};

use crate::registry::{DomainParser, ParseResult};

// ── Parser ───────────────────────────────────────────────────────────────────

pub struct DocumentParser;

impl DomainParser for DocumentParser {
    fn domain_id(&self) -> &'static str {
        "document"
    }

    fn supported_extensions(&self) -> &[&'static str] {
        &[".md", ".mdx", ".html", ".htm", ".txt", ".rst"]
    }

    fn parse(
        &self,
        file_path: &str,
        source: &[u8],
        _config: &TrevecConfig,
    ) -> Result<ParseResult> {
        let text = String::from_utf8_lossy(source);
        let path_lower = file_path.to_lowercase();

        if path_lower.ends_with(".html") || path_lower.ends_with(".htm") {
            parse_html(file_path, &text)
        } else if path_lower.ends_with(".rst") {
            parse_rst(file_path, &text)
        } else if path_lower.ends_with(".txt") {
            parse_plain_text(file_path, &text)
        } else {
            // .md, .mdx — default to markdown
            parse_markdown(file_path, &text)
        }
    }
}

// ── ID Generation ────────────────────────────────────────────────────────────

/// Generate a deterministic ID from a string using blake3.
fn make_id(input: &str) -> String {
    let hash = blake3::hash(input.as_bytes());
    hash.to_hex()[..32].to_string()
}

// ── Markdown Parsing ─────────────────────────────────────────────────────────

/// Internal representation of a parsed section during markdown parsing.
struct MdSection {
    heading: String,
    depth: usize,
    start_byte: usize,
    start_line: usize,
    /// Paragraph text accumulated under this heading.
    paragraphs: Vec<ParagraphSpan>,
    /// Links found within this section.
    links: Vec<ExtractedLink>,
    /// Identifiers extracted from code blocks.
    code_identifiers: Vec<String>,
}

struct ParagraphSpan {
    text: String,
    start_line: usize,
    start_byte: usize,
    end_byte: usize,
}

#[derive(Debug, Clone)]
struct ExtractedLink {
    text: String,
    url: String,
}

fn parse_markdown(file_path: &str, source: &str) -> Result<ParseResult> {
    static HEADING_RE: OnceLock<Regex> = OnceLock::new();
    static LINK_RE: OnceLock<Regex> = OnceLock::new();
    static CODE_BLOCK_RE: OnceLock<Regex> = OnceLock::new();
    static IDENT_RE: OnceLock<Regex> = OnceLock::new();

    let heading_re = HEADING_RE.get_or_init(|| Regex::new(r"^(#{1,6})\s+(.+)$").unwrap());
    let link_re = LINK_RE.get_or_init(|| Regex::new(r"\[([^\]]+)\]\(([^)]+)\)").unwrap());
    let code_block_re =
        CODE_BLOCK_RE.get_or_init(|| Regex::new(r"(?s)```[^\n]*\n(.*?)```").unwrap());
    let ident_re = IDENT_RE.get_or_init(|| Regex::new(r"\b[a-zA-Z_][a-zA-Z0-9_]{2,}\b").unwrap());

    // First pass: extract code block identifiers per byte range
    let code_blocks: Vec<(usize, usize, Vec<String>)> = code_block_re
        .captures_iter(source)
        .filter_map(|cap| {
            let full = cap.get(0)?;
            let body = cap.get(1)?;
            let idents: Vec<String> = ident_re
                .find_iter(body.as_str())
                .map(|m| m.as_str().to_string())
                .collect::<std::collections::HashSet<_>>()
                .into_iter()
                .take(30)
                .collect();
            Some((full.start(), full.end(), idents))
        })
        .collect();

    // Second pass: split into sections by headings
    let mut sections: Vec<MdSection> = Vec::new();
    let mut current_para_lines: Vec<&str> = Vec::new();
    let mut para_start_line: usize = 0;
    let mut para_start_byte: usize = 0;

    let mut byte_offset: usize = 0;
    let lines: Vec<&str> = source.lines().collect();

    for (line_idx, line) in lines.iter().enumerate() {
        let line_start_byte = byte_offset;
        // advance byte offset past this line + newline
        byte_offset += line.len();
        if byte_offset < source.len() {
            // account for the newline character(s)
            if source.as_bytes().get(byte_offset) == Some(&b'\r') {
                byte_offset += 1;
            }
            if source.as_bytes().get(byte_offset) == Some(&b'\n') {
                byte_offset += 1;
            }
        }

        if let Some(cap) = heading_re.captures(line) {
            // Flush current paragraph to the current section
            flush_paragraph(&mut sections, &mut current_para_lines, para_start_line, para_start_byte, line_start_byte);

            let depth = cap.get(1).unwrap().as_str().len();
            let heading_text = cap.get(2).unwrap().as_str().trim().to_string();

            sections.push(MdSection {
                heading: heading_text,
                depth,
                start_byte: line_start_byte,
                start_line: line_idx,
                paragraphs: Vec::new(),
                links: Vec::new(),
                code_identifiers: Vec::new(),
            });
            para_start_line = line_idx + 1;
            para_start_byte = byte_offset;
        } else if line.trim().is_empty() {
            // Blank line terminates a paragraph
            flush_paragraph(&mut sections, &mut current_para_lines, para_start_line, para_start_byte, line_start_byte);
            para_start_line = line_idx + 1;
            para_start_byte = byte_offset;
        } else {
            if current_para_lines.is_empty() {
                para_start_line = line_idx;
                para_start_byte = line_start_byte;
            }
            current_para_lines.push(line);
        }
    }

    // Flush final paragraph
    flush_paragraph(&mut sections, &mut current_para_lines, para_start_line, para_start_byte, byte_offset);

    // If no sections were found, create a root section for the whole document
    if sections.is_empty() && !source.trim().is_empty() {
        let all_paras = collect_paragraphs(source);
        sections.push(MdSection {
            heading: file_path
                .rsplit(['/', '\\'])
                .next()
                .unwrap_or(file_path)
                .to_string(),
            depth: 0,
            start_byte: 0,
            start_line: 0,
            paragraphs: all_paras,
            links: Vec::new(),
            code_identifiers: Vec::new(),
        });
    }

    // Extract links and code identifiers for each section
    for section in &mut sections {
        // Gather links from section heading + paragraphs
        let section_text = {
            let mut t = section.heading.clone();
            for p in &section.paragraphs {
                t.push(' ');
                t.push_str(&p.text);
            }
            t
        };
        for cap in link_re.captures_iter(&section_text) {
            if let (Some(text), Some(url)) = (cap.get(1), cap.get(2)) {
                section.links.push(ExtractedLink {
                    text: text.as_str().to_string(),
                    url: url.as_str().to_string(),
                });
            }
        }

        // Find code block identifiers that fall within this section's byte range
        let section_end = section
            .paragraphs
            .last()
            .map(|p| p.end_byte)
            .unwrap_or(section.start_byte + section.heading.len());
        for (cb_start, cb_end, idents) in &code_blocks {
            if *cb_start >= section.start_byte && *cb_end <= section_end {
                section.code_identifiers.extend(idents.iter().cloned());
            }
        }
    }

    build_nodes_and_edges(file_path, &sections)
}

/// Flush accumulated paragraph lines into the current section.
fn flush_paragraph(
    sections: &mut [MdSection],
    lines: &mut Vec<&str>,
    start_line: usize,
    start_byte: usize,
    end_byte: usize,
) {
    if lines.is_empty() {
        return;
    }
    let text = lines.join(" ").trim().to_string();
    if !text.is_empty() {
        if let Some(section) = sections.last_mut() {
            section.paragraphs.push(ParagraphSpan {
                text,
                start_line,
                start_byte,
                end_byte,
            });
        }
    }
    lines.clear();
}

/// Collect paragraphs from raw text (for files with no headings).
fn collect_paragraphs(source: &str) -> Vec<ParagraphSpan> {
    let mut paragraphs = Vec::new();
    let mut current_lines: Vec<&str> = Vec::new();
    let mut para_start_line: usize = 0;
    let mut para_start_byte: usize = 0;
    let mut byte_offset: usize = 0;

    for (line_idx, line) in source.lines().enumerate() {
        let line_start_byte = byte_offset;
        byte_offset += line.len();
        if byte_offset < source.len() {
            if source.as_bytes().get(byte_offset) == Some(&b'\r') {
                byte_offset += 1;
            }
            if source.as_bytes().get(byte_offset) == Some(&b'\n') {
                byte_offset += 1;
            }
        }

        if line.trim().is_empty() {
            if !current_lines.is_empty() {
                let text = current_lines.join(" ").trim().to_string();
                if !text.is_empty() {
                    paragraphs.push(ParagraphSpan {
                        text,
                        start_line: para_start_line,
                        start_byte: para_start_byte,
                        end_byte: line_start_byte,
                    });
                }
                current_lines.clear();
            }
            para_start_line = line_idx + 1;
            para_start_byte = byte_offset;
        } else {
            if current_lines.is_empty() {
                para_start_line = line_idx;
                para_start_byte = line_start_byte;
            }
            current_lines.push(line);
        }
    }

    // Flush remaining
    if !current_lines.is_empty() {
        let text = current_lines.join(" ").trim().to_string();
        if !text.is_empty() {
            paragraphs.push(ParagraphSpan {
                text,
                start_line: para_start_line,
                start_byte: para_start_byte,
                end_byte: byte_offset,
            });
        }
    }

    paragraphs
}

/// Convert parsed sections into UniversalNodes and UniversalEdges.
fn build_nodes_and_edges(file_path: &str, sections: &[MdSection]) -> Result<ParseResult> {
    let mut nodes = Vec::new();
    let mut edges = Vec::new();

    // Track section hierarchy for nesting edges: stack of (depth, node_id)
    let mut section_stack: Vec<(usize, String)> = Vec::new();

    for (sec_idx, section) in sections.iter().enumerate() {
        let section_node_id = make_id(&format!(
            "doc:section:{}:{}:{}",
            file_path, sec_idx, section.heading
        ));

        let mut section_identifiers: Vec<String> = extract_doc_keywords(&section.heading);
        section_identifiers.extend(section.code_identifiers.iter().cloned());
        section_identifiers.sort();
        section_identifiers.dedup();
        section_identifiers.truncate(30);

        let section_bm25 = format!(
            "{} {} {}",
            file_path,
            section.heading,
            section_identifiers.join(" ")
        );

        let mut attrs = HashMap::new();
        attrs.insert(
            "depth".into(),
            AttributeValue::Int(section.depth as i64),
        );

        let section_end_byte = section
            .paragraphs
            .last()
            .map(|p| p.end_byte)
            .unwrap_or(section.start_byte + section.heading.len());

        let section_end_line = section
            .paragraphs
            .last()
            .map(|p| p.start_line + p.text.lines().count())
            .unwrap_or(section.start_line + 1);

        nodes.push(UniversalNode {
            id: section_node_id.clone(),
            kind: UniversalKind::Section,
            domain: DomainTag::Document,
            label: section.heading.clone(),
            file_path: file_path.to_string(),
            span: Some(trevec_core::Span {
                start_line: section.start_line,
                start_col: 0,
                end_line: section_end_line,
                end_col: 0,
                start_byte: section.start_byte,
                end_byte: section_end_byte,
            }),
            signature: Some(format!(
                "h{}: {}",
                section.depth, section.heading
            )),
            doc_comment: None,
            identifiers: section_identifiers,
            bm25_text: section_bm25,
            symbol_vec: None,
            ast_hash: Some(make_id(&format!(
                "hash:{}:{}",
                file_path,
                &section.heading
            ))),
            temporal: None,
            attributes: attrs,
            intent_summary: None,
        });

        // Build nesting edges: pop stack until we find a parent with lower depth
        while let Some((parent_depth, _)) = section_stack.last() {
            if *parent_depth >= section.depth {
                section_stack.pop();
            } else {
                break;
            }
        }
        if let Some((_, parent_id)) = section_stack.last() {
            edges.push(UniversalEdge::new(
                parent_id.clone(),
                section_node_id.clone(),
                UniversalEdgeType::Contain,
                Confidence::Certain,
            ));
        }
        section_stack.push((section.depth, section_node_id.clone()));

        // Create Fact nodes for paragraphs
        for (para_idx, para) in section.paragraphs.iter().enumerate() {
            let fact_node_id = make_id(&format!(
                "doc:fact:{}:{}:{}",
                file_path, sec_idx, para_idx
            ));

            let fact_label = if para.text.len() > 120 {
                format!("{}...", &para.text[..120])
            } else {
                para.text.clone()
            };

            let fact_identifiers = extract_doc_keywords(&para.text);
            let fact_bm25 = format!("{} {} {}", file_path, para.text, fact_identifiers.join(" "));

            nodes.push(UniversalNode {
                id: fact_node_id.clone(),
                kind: UniversalKind::Fact,
                domain: DomainTag::Document,
                label: fact_label,
                file_path: file_path.to_string(),
                span: Some(trevec_core::Span {
                    start_line: para.start_line,
                    start_col: 0,
                    end_line: para.start_line + para.text.lines().count().max(1),
                    end_col: 0,
                    start_byte: para.start_byte,
                    end_byte: para.end_byte,
                }),
                signature: None,
                doc_comment: None,
                identifiers: fact_identifiers,
                bm25_text: fact_bm25,
                symbol_vec: None,
                ast_hash: None,
                temporal: None,
                attributes: HashMap::new(),
                intent_summary: None,
            });

            // Section --Contain--> Fact
            edges.push(UniversalEdge::new(
                section_node_id.clone(),
                fact_node_id.clone(),
                UniversalEdgeType::Contain,
                Confidence::Certain,
            ));
        }

        // Create Citation nodes for links
        for (link_idx, link) in section.links.iter().enumerate() {
            let citation_node_id = make_id(&format!(
                "doc:cite:{}:{}:{}:{}",
                file_path, sec_idx, link_idx, link.url
            ));

            // Avoid duplicate citation nodes for the same URL within the file
            if nodes.iter().any(|n| n.id == citation_node_id) {
                // Just add a reference edge from the nearest fact
                if let Some(fact_node) = nodes
                    .iter()
                    .rev()
                    .find(|n| n.kind == UniversalKind::Fact)
                {
                    edges.push(UniversalEdge::new(
                        fact_node.id.clone(),
                        citation_node_id,
                        UniversalEdgeType::Reference,
                        Confidence::Likely,
                    ));
                }
                continue;
            }

            let mut cite_attrs = HashMap::new();
            cite_attrs.insert("url".into(), AttributeValue::String(link.url.clone()));

            nodes.push(UniversalNode {
                id: citation_node_id.clone(),
                kind: UniversalKind::Citation,
                domain: DomainTag::Document,
                label: link.text.clone(),
                file_path: file_path.to_string(),
                span: None,
                signature: Some(format!("[{}]({})", link.text, link.url)),
                doc_comment: None,
                identifiers: vec![link.text.clone()],
                bm25_text: format!("{} {} {}", file_path, link.text, link.url),
                symbol_vec: None,
                ast_hash: None,
                temporal: None,
                attributes: cite_attrs,
                intent_summary: None,
            });

            // Find the fact node that most likely contains this link, or use section
            let referencing_node_id = nodes
                .iter()
                .rev()
                .find(|n| {
                    n.kind == UniversalKind::Fact
                        && n.file_path == file_path
                        && n.label.contains(&link.text)
                })
                .map(|n| n.id.clone())
                .unwrap_or_else(|| section_node_id.clone());

            edges.push(UniversalEdge::new(
                referencing_node_id,
                citation_node_id,
                UniversalEdgeType::Reference,
                Confidence::Likely,
            ));
        }
    }

    Ok(ParseResult { nodes, edges })
}

// ── HTML Parsing ─────────────────────────────────────────────────────────────

fn parse_html(file_path: &str, source: &str) -> Result<ParseResult> {
    static HEADING_RES: OnceLock<Vec<(usize, Regex)>> = OnceLock::new();
    static PARA_RE: OnceLock<Regex> = OnceLock::new();
    static LINK_RE: OnceLock<Regex> = OnceLock::new();
    static STRIP_TAGS_RE: OnceLock<Regex> = OnceLock::new();

    // One regex per heading level (h1-h6) since `regex` crate has no backreferences
    let heading_res = HEADING_RES.get_or_init(|| {
        (1..=6usize)
            .map(|level| {
                let pat = format!("(?si)<h{level}[^>]*>(.*?)</h{level}>");
                (level, Regex::new(&pat).unwrap())
            })
            .collect()
    });
    let para_re =
        PARA_RE.get_or_init(|| Regex::new(r"(?si)<p[^>]*>(.*?)</p>").unwrap());
    let link_re =
        LINK_RE.get_or_init(|| Regex::new(r#"(?si)<a[^>]+href="([^"]*)"[^>]*>(.*?)</a>"#).unwrap());
    let strip_tags_re = STRIP_TAGS_RE.get_or_init(|| Regex::new(r"<[^>]+>").unwrap());

    let strip = |s: &str| -> String {
        strip_tags_re.replace_all(s, "").trim().to_string()
    };

    let mut sections: Vec<MdSection> = Vec::new();

    // Extract headings and their positions (all levels), then sort by source position
    struct RawHeading {
        depth: usize,
        text: String,
        start_byte: usize,
    }
    let mut all_headings: Vec<RawHeading> = Vec::new();
    for (level, re) in heading_res.iter() {
        for cap in re.captures_iter(source) {
            let heading_text = strip(cap.get(1).unwrap().as_str());
            let start_byte = cap.get(0).unwrap().start();
            all_headings.push(RawHeading {
                depth: *level,
                text: heading_text,
                start_byte,
            });
        }
    }
    all_headings.sort_by_key(|h| h.start_byte);

    for h in &all_headings {
        let start_line = source[..h.start_byte].lines().count().saturating_sub(1);
        sections.push(MdSection {
            heading: h.text.clone(),
            depth: h.depth,
            start_byte: h.start_byte,
            start_line,
            paragraphs: Vec::new(),
            links: Vec::new(),
            code_identifiers: Vec::new(),
        });
    }

    // Assign paragraphs to nearest preceding heading
    for cap in para_re.captures_iter(source) {
        let para_text = strip(cap.get(1).unwrap().as_str());
        if para_text.is_empty() {
            continue;
        }
        let para_start = cap.get(0).unwrap().start();
        let para_end = cap.get(0).unwrap().end();
        let para_line = source[..para_start].lines().count().saturating_sub(1);

        // Find the section this paragraph belongs to
        let target = sections
            .iter_mut()
            .rev()
            .find(|s| s.start_byte <= para_start);

        if let Some(section) = target {
            section.paragraphs.push(ParagraphSpan {
                text: para_text,
                start_line: para_line,
                start_byte: para_start,
                end_byte: para_end,
            });
        }
    }

    // Extract links
    for cap in link_re.captures_iter(source) {
        if let (Some(url), Some(text)) = (cap.get(1), cap.get(2)) {
            let link_start = cap.get(0).unwrap().start();
            let link_text = strip(text.as_str());
            let link_url = url.as_str().to_string();

            if link_text.is_empty() {
                continue;
            }

            let target = sections
                .iter_mut()
                .rev()
                .find(|s| s.start_byte <= link_start);

            if let Some(section) = target {
                section.links.push(ExtractedLink {
                    text: link_text,
                    url: link_url,
                });
            }
        }
    }

    // If no headings found, create root section
    if sections.is_empty() && !source.trim().is_empty() {
        let mut root = MdSection {
            heading: file_path
                .rsplit(['/', '\\'])
                .next()
                .unwrap_or(file_path)
                .to_string(),
            depth: 0,
            start_byte: 0,
            start_line: 0,
            paragraphs: Vec::new(),
            links: Vec::new(),
            code_identifiers: Vec::new(),
        };

        // Collect all paragraphs
        for cap in para_re.captures_iter(source) {
            let para_text = strip(cap.get(1).unwrap().as_str());
            if !para_text.is_empty() {
                let start = cap.get(0).unwrap().start();
                let end = cap.get(0).unwrap().end();
                root.paragraphs.push(ParagraphSpan {
                    text: para_text,
                    start_line: source[..start].lines().count().saturating_sub(1),
                    start_byte: start,
                    end_byte: end,
                });
            }
        }

        // Collect all links
        for cap in link_re.captures_iter(source) {
            if let (Some(url), Some(text)) = (cap.get(1), cap.get(2)) {
                let lt = strip(text.as_str());
                if !lt.is_empty() {
                    root.links.push(ExtractedLink {
                        text: lt,
                        url: url.as_str().to_string(),
                    });
                }
            }
        }

        sections.push(root);
    }

    build_nodes_and_edges(file_path, &sections)
}

// ── reStructuredText Parsing ─────────────────────────────────────────────────

fn parse_rst(file_path: &str, source: &str) -> Result<ParseResult> {
    static RST_HEADING_RE: OnceLock<Regex> = OnceLock::new();
    static RST_LINK_RE: OnceLock<Regex> = OnceLock::new();

    // RST headings: a line followed by a line of consistent adornment characters
    let _rst_heading_re = RST_HEADING_RE
        .get_or_init(|| Regex::new(r#"^(.+)\n([=\-~^+*]{3,})$"#).unwrap());
    let rst_link_re: &Regex =
        RST_LINK_RE.get_or_init(|| Regex::new(r"`([^`]+)\s*<([^>]+)>`_").unwrap());

    // RST heading depth by adornment character (common convention)
    let rst_depth = |ch: char| -> usize {
        match ch {
            '=' => 1,
            '-' => 2,
            '~' => 3,
            '^' => 4,
            _ => 3,
        }
    };

    let mut sections: Vec<MdSection> = Vec::new();
    let mut current_para_lines: Vec<&str> = Vec::new();
    let mut para_start_line: usize = 0;
    let mut para_start_byte: usize = 0;
    let mut byte_offset: usize = 0;

    let lines: Vec<&str> = source.lines().collect();
    let mut skip_next = false;

    for (line_idx, line) in lines.iter().enumerate() {
        let line_start_byte = byte_offset;
        byte_offset += line.len();
        if byte_offset < source.len() {
            if source.as_bytes().get(byte_offset) == Some(&b'\r') {
                byte_offset += 1;
            }
            if source.as_bytes().get(byte_offset) == Some(&b'\n') {
                byte_offset += 1;
            }
        }

        if skip_next {
            skip_next = false;
            para_start_line = line_idx + 1;
            para_start_byte = byte_offset;
            continue;
        }

        // Check if next line is an adornment line (RST heading)
        if line_idx + 1 < lines.len() {
            let next_line = lines[line_idx + 1];
            let trimmed = next_line.trim();
            if !trimmed.is_empty()
                && !line.trim().is_empty()
                && trimmed.len() >= 3
                && trimmed.chars().all(|c| c == trimmed.chars().next().unwrap_or(' '))
                && "=-~^\"'`#+*".contains(trimmed.chars().next().unwrap_or(' '))
            {
                // This is a heading
                flush_paragraph(
                    &mut sections,
                    &mut current_para_lines,
                    para_start_line,
                    para_start_byte,
                    line_start_byte,
                );

                let depth = rst_depth(trimmed.chars().next().unwrap());
                sections.push(MdSection {
                    heading: line.trim().to_string(),
                    depth,
                    start_byte: line_start_byte,
                    start_line: line_idx,
                    paragraphs: Vec::new(),
                    links: Vec::new(),
                    code_identifiers: Vec::new(),
                });
                skip_next = true;
                continue;
            }
        }

        if line.trim().is_empty() {
            flush_paragraph(
                &mut sections,
                &mut current_para_lines,
                para_start_line,
                para_start_byte,
                line_start_byte,
            );
            para_start_line = line_idx + 1;
            para_start_byte = byte_offset;
        } else {
            if current_para_lines.is_empty() {
                para_start_line = line_idx;
                para_start_byte = line_start_byte;
            }
            current_para_lines.push(line);
        }
    }

    flush_paragraph(
        &mut sections,
        &mut current_para_lines,
        para_start_line,
        para_start_byte,
        byte_offset,
    );

    // Extract RST links for each section
    for section in &mut sections {
        let section_text = {
            let mut t = section.heading.clone();
            for p in &section.paragraphs {
                t.push(' ');
                t.push_str(&p.text);
            }
            t
        };
        for cap in rst_link_re.captures_iter(&section_text) {
            if let (Some(text), Some(url)) = (cap.get(1), cap.get(2)) {
                section.links.push(ExtractedLink {
                    text: text.as_str().trim().to_string(),
                    url: url.as_str().to_string(),
                });
            }
        }
    }

    // If no sections, create root
    if sections.is_empty() && !source.trim().is_empty() {
        let all_paras = collect_paragraphs(source);
        sections.push(MdSection {
            heading: file_path
                .rsplit(['/', '\\'])
                .next()
                .unwrap_or(file_path)
                .to_string(),
            depth: 0,
            start_byte: 0,
            start_line: 0,
            paragraphs: all_paras,
            links: Vec::new(),
            code_identifiers: Vec::new(),
        });
    }

    build_nodes_and_edges(file_path, &sections)
}

// ── Plain Text Parsing ───────────────────────────────────────────────────────

fn parse_plain_text(file_path: &str, source: &str) -> Result<ParseResult> {
    // Plain text: no headings, just paragraphs separated by blank lines.
    // Create a single root section containing all paragraphs.
    let paragraphs = collect_paragraphs(source);
    let sections = vec![MdSection {
        heading: file_path
            .rsplit(['/', '\\'])
            .next()
            .unwrap_or(file_path)
            .to_string(),
        depth: 0,
        start_byte: 0,
        start_line: 0,
        paragraphs,
        links: Vec::new(),
        code_identifiers: Vec::new(),
    }];

    build_nodes_and_edges(file_path, &sections)
}

// ── Keyword Extraction ───────────────────────────────────────────────────────

/// Extract significant keywords from document text (similar to conversation parser).
fn extract_doc_keywords(text: &str) -> Vec<String> {
    static STOP_WORDS: OnceLock<std::collections::HashSet<&str>> = OnceLock::new();
    let stop_words = STOP_WORDS.get_or_init(|| {
        [
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "have", "has",
            "had", "do", "does", "did", "will", "would", "shall", "should", "may", "might",
            "must", "can", "could", "i", "you", "he", "she", "it", "we", "they", "me", "him",
            "her", "us", "them", "my", "your", "his", "its", "our", "their", "this", "that",
            "these", "those", "what", "which", "who", "whom", "when", "where", "why", "how",
            "not", "no", "nor", "but", "or", "and", "if", "then", "else", "so", "just", "too",
            "very", "really", "also", "of", "in", "on", "at", "to", "for", "with", "by",
            "from", "about", "into", "through", "during", "before", "after", "above", "below",
            "up", "down", "out", "off", "over", "under", "again", "further", "once", "here",
            "there", "all", "both", "each", "more", "most", "other", "some", "such", "than",
            "as", "use", "using",
        ]
        .into_iter()
        .collect()
    });

    text.split_whitespace()
        .map(|w| {
            w.trim_matches(|c: char| !c.is_alphanumeric())
                .to_lowercase()
        })
        .filter(|w| w.len() > 2 && !stop_words.contains(w.as_str()))
        .take(20)
        .collect()
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> TrevecConfig {
        TrevecConfig::default()
    }

    // ── 1. Basic markdown heading extraction ─────────────────────────────

    #[test]
    fn test_markdown_headings_produce_section_nodes() {
        let md = "# Introduction\n\nSome text.\n\n## Details\n\nMore text.\n";
        let result = parse_markdown("doc.md", md).unwrap();

        let sections: Vec<_> = result
            .nodes
            .iter()
            .filter(|n| n.kind == UniversalKind::Section)
            .collect();
        assert_eq!(sections.len(), 2);
        assert_eq!(sections[0].label, "Introduction");
        assert_eq!(sections[1].label, "Details");

        // Check depth attributes
        assert_eq!(
            sections[0].attributes.get("depth"),
            Some(&AttributeValue::Int(1))
        );
        assert_eq!(
            sections[1].attributes.get("depth"),
            Some(&AttributeValue::Int(2))
        );
    }

    // ── 2. Paragraphs produce Fact nodes ─────────────────────────────────

    #[test]
    fn test_markdown_paragraphs_produce_fact_nodes() {
        let md = "# Title\n\nFirst paragraph here.\n\nSecond paragraph here.\n";
        let result = parse_markdown("doc.md", md).unwrap();

        let facts: Vec<_> = result
            .nodes
            .iter()
            .filter(|n| n.kind == UniversalKind::Fact)
            .collect();
        assert_eq!(facts.len(), 2);
        assert!(facts[0].label.contains("First paragraph"));
        assert!(facts[1].label.contains("Second paragraph"));
    }

    // ── 3. Section --Contain--> Fact edges ───────────────────────────────

    #[test]
    fn test_section_contain_fact_edges() {
        let md = "# Title\n\nA paragraph.\n";
        let result = parse_markdown("doc.md", md).unwrap();

        let section_id = result
            .nodes
            .iter()
            .find(|n| n.kind == UniversalKind::Section)
            .unwrap()
            .id
            .clone();
        let fact_id = result
            .nodes
            .iter()
            .find(|n| n.kind == UniversalKind::Fact)
            .unwrap()
            .id
            .clone();

        let contain_edges: Vec<_> = result
            .edges
            .iter()
            .filter(|e| {
                e.edge_type == UniversalEdgeType::Contain
                    && e.src_id == section_id
                    && e.dst_id == fact_id
            })
            .collect();
        assert_eq!(contain_edges.len(), 1);
        assert_eq!(contain_edges[0].confidence, Confidence::Certain);
    }

    // ── 4. Nested sections produce containment edges ─────────────────────

    #[test]
    fn test_nested_section_containment() {
        let md = "# Chapter 1\n\n## Section 1.1\n\nText.\n\n## Section 1.2\n\nText.\n";
        let result = parse_markdown("doc.md", md).unwrap();

        let sections: Vec<_> = result
            .nodes
            .iter()
            .filter(|n| n.kind == UniversalKind::Section)
            .collect();
        assert_eq!(sections.len(), 3);

        // The h1 should contain the two h2 sections
        let chapter_id = &sections[0].id;
        let nesting_edges: Vec<_> = result
            .edges
            .iter()
            .filter(|e| {
                e.edge_type == UniversalEdgeType::Contain
                    && e.src_id == *chapter_id
                    && result
                        .nodes
                        .iter()
                        .any(|n| n.id == e.dst_id && n.kind == UniversalKind::Section)
            })
            .collect();
        assert_eq!(nesting_edges.len(), 2);
    }

    // ── 5. Links produce Citation nodes with Reference edges ─────────────

    #[test]
    fn test_markdown_links_produce_citations() {
        let md = "# Resources\n\nSee [Rust Book](https://doc.rust-lang.org/book/) for details.\n";
        let result = parse_markdown("doc.md", md).unwrap();

        let citations: Vec<_> = result
            .nodes
            .iter()
            .filter(|n| n.kind == UniversalKind::Citation)
            .collect();
        assert_eq!(citations.len(), 1);
        assert_eq!(citations[0].label, "Rust Book");
        assert_eq!(
            citations[0].attributes.get("url"),
            Some(&AttributeValue::String(
                "https://doc.rust-lang.org/book/".to_string()
            ))
        );

        // Reference edge should exist
        let ref_edges: Vec<_> = result
            .edges
            .iter()
            .filter(|e| e.edge_type == UniversalEdgeType::Reference)
            .collect();
        assert_eq!(ref_edges.len(), 1);
    }

    // ── 6. Code blocks extract identifiers ───────────────────────────────

    #[test]
    fn test_code_block_identifiers() {
        let md = "# API\n\n```rust\nfn authenticate(user: &User) -> bool {\n    validate_token(user.token)\n}\n```\n";
        let result = parse_markdown("doc.md", md).unwrap();

        let section = result
            .nodes
            .iter()
            .find(|n| n.kind == UniversalKind::Section)
            .unwrap();

        // Should have extracted identifiers from the code block
        let has_code_idents = section
            .identifiers
            .iter()
            .any(|id| id == "authenticate" || id == "validate_token" || id == "user");
        assert!(
            has_code_idents,
            "Expected code identifiers, got: {:?}",
            section.identifiers
        );
    }

    // ── 7. All nodes tagged with Document domain ─────────────────────────

    #[test]
    fn test_all_nodes_document_domain() {
        let md = "# Title\n\nParagraph with [a link](http://example.com).\n";
        let result = parse_markdown("doc.md", md).unwrap();

        assert!(!result.nodes.is_empty());
        for node in &result.nodes {
            assert_eq!(
                node.domain,
                DomainTag::Document,
                "Node {} has domain {:?}, expected Document",
                node.id,
                node.domain
            );
        }
    }

    // ── 8. DomainParser trait implementation ──────────────────────────────

    #[test]
    fn test_document_parser_trait() {
        let parser = DocumentParser;
        assert_eq!(parser.domain_id(), "document");

        let exts = parser.supported_extensions();
        assert!(exts.contains(&".md"));
        assert!(exts.contains(&".mdx"));
        assert!(exts.contains(&".html"));
        assert!(exts.contains(&".htm"));
        assert!(exts.contains(&".txt"));
        assert!(exts.contains(&".rst"));

        assert!(parser.can_parse("README.md", &[]));
        assert!(parser.can_parse("index.html", &[]));
        assert!(parser.can_parse("notes.txt", &[]));
        assert!(!parser.can_parse("code.rs", &[]));
    }

    // ── 9. HTML parsing extracts headings and paragraphs ─────────────────

    #[test]
    fn test_html_parsing() {
        let html = r#"<html><body>
<h1>Welcome</h1>
<p>This is the intro.</p>
<h2>Details</h2>
<p>Some details here.</p>
<p>More <a href="https://example.com">info</a>.</p>
</body></html>"#;

        let parser = DocumentParser;
        let config = default_config();
        let result = parser.parse("page.html", html.as_bytes(), &config).unwrap();

        let sections: Vec<_> = result
            .nodes
            .iter()
            .filter(|n| n.kind == UniversalKind::Section)
            .collect();
        assert_eq!(sections.len(), 2);
        assert_eq!(sections[0].label, "Welcome");
        assert_eq!(sections[1].label, "Details");

        let facts: Vec<_> = result
            .nodes
            .iter()
            .filter(|n| n.kind == UniversalKind::Fact)
            .collect();
        assert!(facts.len() >= 2);

        let citations: Vec<_> = result
            .nodes
            .iter()
            .filter(|n| n.kind == UniversalKind::Citation)
            .collect();
        assert_eq!(citations.len(), 1);
        assert_eq!(citations[0].label, "info");
    }

    // ── 10. Plain text produces root section with fact nodes ─────────────

    #[test]
    fn test_plain_text_parsing() {
        let text = "First paragraph of the document.\n\nSecond paragraph with more info.\n\nThird paragraph.\n";

        let parser = DocumentParser;
        let config = default_config();
        let result = parser.parse("notes.txt", text.as_bytes(), &config).unwrap();

        let sections: Vec<_> = result
            .nodes
            .iter()
            .filter(|n| n.kind == UniversalKind::Section)
            .collect();
        assert_eq!(sections.len(), 1, "Plain text should have one root section");
        assert_eq!(sections[0].label, "notes.txt");

        let facts: Vec<_> = result
            .nodes
            .iter()
            .filter(|n| n.kind == UniversalKind::Fact)
            .collect();
        assert_eq!(facts.len(), 3);
    }

    // ── 11. Deterministic IDs via blake3 ─────────────────────────────────

    #[test]
    fn test_deterministic_ids() {
        let md = "# Title\n\nSome text.\n";

        let r1 = parse_markdown("doc.md", md).unwrap();
        let r2 = parse_markdown("doc.md", md).unwrap();

        assert_eq!(r1.nodes.len(), r2.nodes.len());
        for (a, b) in r1.nodes.iter().zip(r2.nodes.iter()) {
            assert_eq!(a.id, b.id, "IDs should be deterministic");
            assert_eq!(a.id.len(), 32, "IDs should be 32 hex chars");
        }
    }

    // ── 12. RST heading parsing ──────────────────────────────────────────

    #[test]
    fn test_rst_parsing() {
        let rst = "Introduction\n============\n\nSome introduction text.\n\nDetails\n-------\n\nDetailed info here.\n";

        let parser = DocumentParser;
        let config = default_config();
        let result = parser.parse("guide.rst", rst.as_bytes(), &config).unwrap();

        let sections: Vec<_> = result
            .nodes
            .iter()
            .filter(|n| n.kind == UniversalKind::Section)
            .collect();
        assert_eq!(sections.len(), 2);
        assert_eq!(sections[0].label, "Introduction");
        assert_eq!(sections[1].label, "Details");

        // Check RST depth convention: = is depth 1, - is depth 2
        assert_eq!(
            sections[0].attributes.get("depth"),
            Some(&AttributeValue::Int(1))
        );
        assert_eq!(
            sections[1].attributes.get("depth"),
            Some(&AttributeValue::Int(2))
        );
    }

    // ── 13. Empty input produces empty result ────────────────────────────

    #[test]
    fn test_empty_input() {
        let result = parse_markdown("empty.md", "").unwrap();
        assert!(result.nodes.is_empty());
        assert!(result.edges.is_empty());
    }

    // ── 14. Span information is populated ────────────────────────────────

    #[test]
    fn test_span_information() {
        let md = "# Title\n\nA paragraph.\n";
        let result = parse_markdown("doc.md", md).unwrap();

        let section = result
            .nodes
            .iter()
            .find(|n| n.kind == UniversalKind::Section)
            .unwrap();
        assert!(section.span.is_some());
        let span = section.span.as_ref().unwrap();
        assert_eq!(span.start_line, 0);
        assert_eq!(span.start_byte, 0);
    }

    // ── 15. BM25 text includes file path ─────────────────────────────────

    #[test]
    fn test_bm25_text_includes_file_path() {
        let md = "# Architecture\n\nThe system uses microservices.\n";
        let result = parse_markdown("docs/arch.md", md).unwrap();

        for node in &result.nodes {
            assert!(
                node.bm25_text.contains("docs/arch.md"),
                "bm25_text should include file path: {}",
                node.bm25_text
            );
        }
    }

    // ── 16. Multiple links in one section ────────────────────────────────

    #[test]
    fn test_multiple_links_in_section() {
        let md = "# Links\n\nSee [Alpha](http://a.com) and [Beta](http://b.com) for more.\n";
        let result = parse_markdown("doc.md", md).unwrap();

        let citations: Vec<_> = result
            .nodes
            .iter()
            .filter(|n| n.kind == UniversalKind::Citation)
            .collect();
        assert_eq!(citations.len(), 2);

        let ref_edges: Vec<_> = result
            .edges
            .iter()
            .filter(|e| e.edge_type == UniversalEdgeType::Reference)
            .collect();
        assert_eq!(ref_edges.len(), 2);
    }
}
