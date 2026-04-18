//! Core domain logic for flowd.
//!
//! This crate defines the trait abstractions and domain types shared across
//! all flowd subsystems: memory, orchestration, and rules enforcement.
//! It contains no I/O framework dependencies -- pure logic only.

pub mod error;
pub mod memory;
pub mod orchestration;
pub mod rules;
pub mod types;
