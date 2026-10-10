#pragma once

// ---------------------------------------------------------------------
// Declarative Attention (DA) control tags
// ---------------------------------------------------------------------
// The model emits control tags in its own output; the server enforces the
// matching attention scope mid-decode:
//   <focus magic_chunks="N">   enter FOCUS: keep chunk N (1-based) + scaffold
//   <local>                   enter LOCAL: scaffold + generated only
//   </focus> / </local>       return to GLOBAL (full attention)
//   <global> / </global>      (re)enter GLOBAL - no-op when already there
// where scaffold = every prompt token outside the chunk/filler ranges
// (system, the question, the injected instruction); the generated tail
// is always attended.
//
// Split out of server_context_impl (2026-10-18): pure string functions -
// the only inputs are the generated text and the current da_mode_t - so
// they are free functions here and unit-testable without a server.

#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

// DA mode state machine (P3): the attention scope a slot is currently
// decoding with. GLOBAL = full context; FOCUS = only the kept chunk(s) +
// scaffold; LOCAL = scaffold + generated only (every chunk excluded).
// Transitions are driven by the model's own tags (see apply_da_tag):
//   <focus magic_chunks="N">  GLOBAL/LOCAL -> FOCUS (keep N)
//   <local>                   GLOBAL/FOCUS -> LOCAL
//   </focus> / </local>       FOCUS/LOCAL  -> GLOBAL (return)
//   <global> / </global>      any          -> GLOBAL (explicit, no-op if there)
enum da_mode_t { DA_MODE_GLOBAL = 0, DA_MODE_FOCUS, DA_MODE_LOCAL };

struct da_tag_t {
    int                  type      = -1; // 0=<focus N>, 1=<local>, 2=</focus>, 3=</local>, 4=<global>, 5=</global>
    std::vector<int32_t> keep_nums;      // chunk numbers (type 0 only), as emitted
    size_t               start     = 0;  // char offset in the scanned text
    size_t               end       = 0;  // char offset just past the closing '>'
};

// Tag-start rule, shared by scan_da_tag / da_tag_hold_len /
// da_tag_inflight (the three must stay in sync - 0602ad47c). S1 +
// S1-CLOSE (line-start only, all six tags): every control tag - the
// two ENTRY tags (the focus and the local opener), the three RETURN
// (close) tags, and the global tag - starts only at the beginning of
// the text or right after a newline. The model emits control tags as
// standalone lines (the da-auto instruction mandates it: "Emit every
// control tag on its own line - a tag quoted mid-line is data, not a
// control tag"), and a literal tag quoted or embedded mid-line (after
// a space, a backtick, or glued to the answer) is data, not a control
// tag. This drops the legacy "after any whitespace" rule and the P3
// glued-return exception for the close tags, which 345642 (123,
// 09-24, DA on) showed still consumed mid-line close quotes - erasing
// them from the model's own reasoning and forcing spurious GLOBAL
// returns (the L2 re-derivation loop). A genuine close emitted
// mid-line now fails open: the tag stays visible, the mode is
// unchanged, and the model re-emits it on its own line per the
// instruction.
bool da_tag_start_allowed(const std::string & text, size_t lt, da_mode_t mode);

// Find the earliest complete DA tag in text[from, size()). Returns
// type < 0 while no tag is closed yet. A full re-scan per token is fine:
// generation is short and the scan starts at the last consumed position.
// A tag must start where da_tag_start_allowed() passes - any of the six
// tags, only at the beginning of the text or right after a newline
// (S1 + S1-CLOSE).
da_tag_t scan_da_tag(const std::string & text, size_t from, da_mode_t mode);

// True while the candidate fragment (a suffix of the generated text
// starting at a tag-start '<') could still grow into a complete DA tag:
// a proper prefix of a fixed tag, or the <focus magic_chunks="N"> head
// plus a number list (digit runs, comma-separated) that is not closed
// yet. A complete tag returns false (apply_da_tag() has consumed it).
bool da_tag_prefix(const std::string & s);

// Number of trailing characters of `unsent` that process_token() must
// hold back from the output stream while the tail could still grow into
// a complete DA tag. Holds from the last tag-start '<' (same start rule
// as scan_da_tag) through the in-progress fragment (da_tag_prefix). This
// covers the full <focus magic_chunks="N> prefix, not just the fixed
// opener: the old opener-only hold dropped to 0 the moment the model
// wrote the opening quote, flushing the tag head + chunk number to the
// client before the closing '>' completed and erased the tag (leaving a
// visible "<focus magic_chunks=\"N" stub). A complete tag is not held
// (apply_da_tag has already consumed it).
size_t da_tag_hold_len(const std::string & unsent, da_mode_t mode);

// True while the generated tail could still grow into a complete DA tag:
// the last tag-start '<' (same start rule as scan_da_tag) opens a
// fragment that is a proper tag prefix. Used by the DA tag gate to hold
// speculative drafting back for as long as a tag is being emitted.
bool da_tag_inflight(const std::string & text, da_mode_t mode);
