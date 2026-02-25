//! Tests verifying that the parser does not stack overflow on deeply nested ASTs.
//!
//! Each test runs the parser in a thread with a constrained stack and feeds it
//! pathological input.  Without the recursion guards in `analysis.rs` the
//! helper functions blow the stack, causing the thread to panic.
//!
//! Stack budget reasoning (debug-mode frames are ~1-2 KB each):
//!
//! - Already-guarded functions (`extract_from_node`, `visit`, etc.) recurse
//!   up to `max_recursion_depth` (default 1024).  1024 × ~2 KB ≈ 2 MB.
//! - The **un-guarded** helpers (before the fix) recurse up to AST depth.
//!   At 10 000 levels that's 10 000 × ~2 KB ≈ 20 MB — far over 4 MB.
//! - After the fix the helpers are either non-recursive (heap-allocated
//!   stack in `find_first_by_kind`) or capped at 1024 depth, so total
//!   stack stays ≈ 2 MB.
//!
//! By using a 4 MB stack with 10 000 levels of nesting we ensure:
//!   guarded code fits  (~2 MB  < 4 MB) ✓
//!   un-guarded overflows (~20 MB > 4 MB) ✗  — test catches the bug
//!   fixed code fits     (~2 MB  < 4 MB) ✓

use super::common::*;
use crate::parser::Language;

const NESTING: usize = 10_000;

/// Build a C function whose single parameter is wrapped in `depth` levels of
/// pointer-declarator nesting:  `void f(int (*(*( … (*param) … ))) { }`
fn deeply_nested_c_declarator(depth: usize) -> String {
    let mut code = String::from("void f(int ");
    for _ in 0..depth {
        code.push_str("(*");
    }
    code.push_str("param");
    for _ in 0..depth {
        code.push(')');
    }
    code.push_str(") { }");
    code
}

/// Build a C++ class that extends a deeply nested template type:
///   `class Foo : public W0<W1<W2< … <Base> … >>> {};`
fn deeply_nested_cpp_inheritance(depth: usize) -> String {
    let mut parent = String::from("Base");
    for i in 0..depth {
        parent = format!("W{}<{}>", i, parent);
    }
    format!("class Foo : public {} {{}};", parent)
}

#[test]
fn test_deeply_nested_c_declarator_no_overflow() {
    let code = deeply_nested_c_declarator(NESTING);
    let result = std::thread::Builder::new()
        .stack_size(8 * 1024 * 1024) // 8 MB
        .spawn(move || {
            parse(&code, Language::C, "test.c");
        })
        .unwrap()
        .join();
    assert!(
        result.is_ok(),
        "Parser stack-overflowed on a deeply nested C declarator"
    );
}

#[test]
fn test_deeply_nested_cpp_inheritance_no_overflow() {
    let code = deeply_nested_cpp_inheritance(NESTING);
    let result = std::thread::Builder::new()
        .stack_size(4 * 1024 * 1024) // 4 MB — extract_from_node returns early for classes, so much less stack needed
        .spawn(move || {
            parse(&code, Language::Cpp, "test.cpp");
        })
        .unwrap()
        .join();
    assert!(
        result.is_ok(),
        "Parser stack-overflowed on deeply nested C++ template inheritance"
    );
}
